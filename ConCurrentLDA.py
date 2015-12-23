import random
import time
from mpi4py import MPI
import numpy
import json
import time

class LdaGibbsSampler:

    def __init__(self, Data, K = 20, P = 10):
        self.D = Data	# input Data
        self.K = K
        self.V = 0
        self.P = P
        self.M = int(len(self.D) * P / 10)
        self.alpha = 2
        self.beta = 0.5
        self.maxIter = 1000
        self.burnIn = 100
        self.sampleLag = 20
        self.ndsum = {} # total number of words in document i
        self.nw = {} 	# number of instances of word i (term?) assigned to topic j
        self.nd = {} 	# number of words in document i assigned to topic j
        self.nwsum = {} # total number of words assigned to topic j
        self.z = {} 	# topic assignments for each word
        self.phisum = {}
        self.numstats = 0.0
        self.thetasum = {}
        self.dict = []
        self.run()
		

    def run(self):
        self.set_V()\
            .set_ND()\
            .set_NW()\
            .set_NWSUM()\
            .set_NDSUM()


    def set_NDSUM(self):
        for i in xrange(self.M):
            self.ndsum[i] = 0.0
        return self


    def set_NWSUM(self):
        for j in xrange(self.K):
            self.nwsum[j] = 0.0
        return self


    def set_NW(self):
        for i in xrange(self.V):
            self.nw[self.dict[i]] = {}
            for j in xrange(self.K):
                self.nw[self.dict[i]][j] = 0.0
        return self


    def set_ND(self):
        for i in xrange(self.M):
            self.nd[i] = {}
            for j in xrange(self.K):
                self.nd[i][j] = 0.0
        return self


    def set_M(self, value = 0):
        self.M = value
        return self


    def set_V(self):
        Set = set()
        for s in self.D[:self.M]:
            Set = Set | set(s)
        self.V = len(Set)
        self.dict = [t for t in Set]
        print "self.V =", self.V
        return self


    def set_K(self, value):
        self.K = value
        return self


    def set_alpha(self, value):
        self.alpha = value
        return self


    def set_beta(self, value):
        self.beta = value
        return self


    def configure(self, iterations, burnIn, sampleLag):
        self.maxIter = iterations
        self.burnIn = burnIn
        self.sampleLag = sampleLag


    def set_thetasum(self):
        for m in xrange(self.M):
            self.thetasum[m] = {}
            for j in xrange(self.K):
                self.thetasum[m][j] = 0.0
        return self


    def set_phisum(self):
        for k in xrange(self.K):
            self.phisum[k] = {}
            for v in xrange(self.V):
                self.phisum[k][self.dict[v]] = 0.0
        return self

    def init_parallel(self, comm):
        node_size = comm.Get_size()
        node_rank = comm.Get_rank()
        self.partition = []
        self.gatherLag = 1
        self.sum = 0
        self.offset = {}

        for i in xrange(self.M):
            N = len(self.D[i])
            self.sum += N
            self.ndsum[i] = N

        partitions = {}
        if node_rank == 0:
            overloads = {}
            for i in xrange(node_size):
                partitions[i] = []
                overloads[i] = 0
            averageload = self.sum / node_size
        
            doclist = range(self.M)
            nodelist = range(node_size)
            curindex = 0
            while(len(doclist) != 0 and len(nodelist) != 0):
                curdoc = random.choice(doclist)
                curnode = nodelist[curindex]
                partitions[curnode].append(curdoc)
                overloads[curnode] += len(self.D[curdoc])
                doclist.remove(curdoc)
                if overloads[curnode] >= averageload:
                    nodelist.remove(curnode)
                    if curindex == len(nodelist):
                        curindex = 0
                else:
                    curindex = (curindex + 1) % len(nodelist)

        partitions = comm.bcast(partitions, root = 0)

        self.partition = partitions[node_rank]
        print node_rank, "has", len(self.partition), "partitions, they are", self.partition
        self.z = numpy.random.rand(self.sum)
        for i in xrange(self.sum):
            self.z[i] = int(self.z[i] * self.K)

        sum = 0
        for m in xrange(len(partitions)):
            for n in partitions[m]:
                self.offset[n] = sum
                sum += len(self.D[n])
	
	
    def gibbs(self, alpha = 2, beta = 0.5):
        comm = MPI.COMM_WORLD
        self.init_parallel(comm)
        if len(self.partition) == 0:
            print "No Data!"
            return

        self.alpha = alpha
        self.beta = beta
        if self.sampleLag > 0:
            self.set_thetasum()\
                .set_phisum()
            self.numstats = 0.0
        #self.initial_state()
		
        self.load_z()
        offset = self.offset[self.partition[0]]
        offsets = comm.allgather(offset)
        print "offsets", offsets
        size = 0
        for i in self.partition:
			size += len(self.D[i])
        sizes = comm.allgather(size)
        print "sizes", sizes, len(self.z), self.sum / comm.Get_size()
		
        for i in xrange(self.maxIter):
            if i % 100 == 0:
                print "iteration", i , time.ctime()			
			
            for m in self.partition:
                for n in xrange(len(self.D[m])):
                    # update the topic distribution of the n-th word in the m-th doc
                    self.z[self.offset[m] + n] = self.sample_full_conditional(m, n)

            #if i % self.gatherLag == 0:
            comm.Allgatherv([self.z[offset:(offset+size)], MPI.DOUBLE],[self.z, sizes, offsets, MPI.DOUBLE])
            self.load_z()
					
            if i > self.burnIn and self.sampleLag > 0 and i % self.sampleLag == 0:
                self.update_params()

        if comm.Get_rank() == 0:
            theta = self.get_theta()
            res_theta = open("theta%d.txt" % self.P, "w")
            res_theta.write(json.dumps(theta,encoding='UTF-8',ensure_ascii=False))
            res_theta.close()
            phi = self.get_phi()
            res_phi = open("phi%d.txt" % self.P, "w")
            res_phi.write(json.dumps(phi,encoding='UTF-8',ensure_ascii=False))
            res_phi.close()


    def sample_full_conditional(self, m, n):
        # m, the index number of doc
        # n, the index number word

        topic = int(self.z[self.offset[m] + n])
        
        # value -1 means word self must be ignored in sampling process
        self.nw[self.D[m][n]][topic] -= 1
        self.nd[m][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[m] -= 1
        
        p = {}
        for k in xrange(self.K):
            p[k] = (self.nw[self.D[m][n]][k] + self.beta) / (self.nwsum[k] + self.V * self.beta) * (self.nd[m][k] + self.alpha) / (self.ndsum[m] + self.K * self.alpha)
        
        for k in xrange(1, len(p)): p[k] += p[k - 1]

        u = random.random() * p[self.K - 1]
        for topic in xrange(len(p)):
            if u < p[topic]: break
        self.nw[self.D[m][n]][topic] += 1  
        self.nd[m][topic] += 1  
        self.nwsum[topic] += 1
        self.ndsum[m] += 1
        return topic


    def update_params(self):
        for m in xrange(self.M):
            for k in xrange(self.K):
                self.thetasum[m][k] += (self.nd[m][k] + self.alpha) / (self.ndsum[m] + self.K * self.alpha)

        for k in xrange(self.K):
            for w in xrange(self.V):
                self.phisum[k][self.dict[w]] += (self.nw[self.dict[w]][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)
        
        self.numstats += 1


    def initial_state(self):
        for m in xrange(self.M):
            N = len(self.D[m]) 
            self.z[m] = []
            for n in xrange(N):
                topic = int(random.random() * self.K)
                self.z[m].append(topic)

                self.nw[self.D[m][n]][topic] = self.nw[self.D[m][n]].get(topic, 0) + 1
                self.nd[m][topic] = self.nd[m].get(topic, 0) + 1
                self.nwsum[topic] = self.nwsum.get(topic, 0) + 1
                
                n += 1

            self.ndsum[m] = N
            
            m += 1

    def load_z(self):
    	self.set_NW()
    	self.set_ND()
    	self.set_NWSUM()
    	for m in xrange(self.M):
    		N = len(self.D[m])
    		for n in xrange(N):
    			topic = int(self.z[self.offset[m] + n])
    			self.nw[self.D[m][n]][topic] = self.nw[self.D[m][n]].get(topic, 0) + 1
    			self.nd[m][topic] = self.nd[m].get(topic, 0) + 1
    			self.nwsum[topic] = self.nwsum.get(topic, 0) + 1
			

    def get_theta(self):
        theta = {}
        for m in xrange(self.M):
            theta[m] = {}
            for k in xrange(self.K):
                theta[m][k] = 0
        if self.sampleLag > 0:
            for m in xrange(self.M):
                for k in xrange(self.K):
                    theta[m][k] = self.thetasum[m][k] / self.numstats
        else:
            for m in xrange(self.M):
                for k in xrange(self.K):
                    theta[m][k] = (self.nd[m][k] + self.alpha) / (self.ndsum[m] + self.K * self.alpha); 
        return theta


    def get_phi(self):
        phi = {}
        for k in xrange(self.K):
            phi[k] = {}
            for v in xrange(self.V):
                phi[k][self.dict[v]] = 0
        if self.sampleLag > 0:
            for k in xrange(self.K):
                for v in xrange(self.V):
                    phi[k][self.dict[v]] = self.phisum[k][self.dict[v]] / self.numstats
        else:
            for k in xrange(self.K):
                for v in xrange(self.V):
                    phi[k][v] = (self.nw[self.dict[k]][v] + self.alpha) / (self.nwsum[k] + self.K * self.alpha); 
        return phi


def loadData(file):
	data = open(file, 'r')
	documents = []
	for line in data:
		linesplit = line[:-1].split(',')
		documents.append(linesplit)
	return documents

def main():
   # training
    documents = loadData("aspectj.data")
    log = open("node%dlog.txt" % MPI.COMM_WORLD.Get_rank(), "w")
   # print documents
    for i in xrange(1, 10): 
        start = time.time()  
        lda = LdaGibbsSampler(documents, 20, i)
        lda.configure(1000, 100, 10)
        lda.gibbs(0.5, 0.01)
        stop = time.time()
        print >> log, "part %d elapsed time %fs" % (i, stop - start)
    log.close()
   # theta = lda.get_theta()
   # print theta
   # testing

if __name__ == '__main__':
	main()
