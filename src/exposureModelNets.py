import math, random
import networkx as nx
import cPickle as pickle
from exposureModelNodes import *
from exposureModelinference import *

class inferenceGraph (nx.Graph):
    '''
    A network class to deal with perturbations on networks.
    It inherits all methos and attributes of networkx graphs.
    '''

    def __init__ (self, init_p=0.5, init_label=0.5):
        '''
        A method to initialise the class.
        Keyword arguments:
        i. init_p : initial value of messages for belief propagation
        i. init_label : the initial value of labels for label propagation
        '''
        ## intialise everything
        ## (including the networkx graph inheritances)
        nx.Graph.__init__(self)
        ## list of observed nodes
        self.observed = []
        ## list of perturbed nodes
        self.perturbed = []
        ## list of unobserved nodes
        self.unobserved = []
        ## nodes lying on dead ends of the net,
        ## i.e. chains of degree 2 nodes,
        ## separated from the rest of the net
        ## by observed unperturbed nodes
        self.pruned_nodes = []
        self.pert_neighbors = []
        self.tag2node = {}
        self.layout = None
        self.name = 'No-name'

        return

    def __repr__ (self):
        '''
        A method to represent the class,
        to see it in command line.
        '''
        
        ## get the number of nodes in the net
        N = len(self.nodes())
        if N>0:
            k_av = float( sum( self.degree().values() ) ) / float( N )
        else:
            k_av = 0.

        ## the string representing the current net
        graph_line = 'Inference graph %s of %d nodes, average degree %g' %(self.name, N, k_av)
        p_size = len(self.perturbed)
        obs_size = len(self.observed)
        pert_obs = len(set(self.observed) & set(self.perturbed))
        unpert_obs = len( set(self.observed) - (set(self.observed) & set(self.perturbed)))
        obs_line = '%d perturbed nodes %d observed nodes (of which %d pert and %d unpert)' %(p_size, obs_size, pert_obs, unpert_obs)

        return '; '.join([graph_line, obs_line])
    
    def __str__ (self):
        '''
        A string method to print the class:
        it simply return self.__repr__
        '''
        return self.__repr__()
    
    def set_bp_lp_messages(self, node, lp_init = 0.5, bp_init=0.5):
        '''
        A method to set up parameters for BP and LP.
        It initialises messages and labels taking into account if
        they're observed or not.
        Mandatory arguments
        i. node : the node to set up
        Keyword arguments:
        i. lp_iinit : the initial value for labels in LP
        ii. bp_init : the initial value of messages in BP
        '''

        ## if the node is observed
        if node.observed:
            ## messages, labels etc.
            ## are initialised to the state value
            ## of the node:
            ## i.e. 1 if perturbed
            ## 0 if unperturbed
            node.bp.psi = node.state
            node.bp.set_messages (self.neighbors(node), psi_value=1.-node.state)
            node.lp.label = node.state
            node.lp.norm = float(len(self.neighbors(node)))
        ## if the node is not observed
        else:
            ## values are initialised at .5
            node.bp.psi = bp_init
            node.bp.set_messages (self.neighbors(node), psi_value=bp_init)
            node.lp.label = lp_init
            node.lp.norm = float(len(self.neighbors(node)))

        return

    def reset_node(self, node, lp_init = 0.5, bp_init=0.5, sp_score = 0.):
        
        '''
        A method to reset the node properties.
        Mandatory arguments
        i. node : the node to set up
        Keyword arguments:
        i. lp_iinit : the initial value for labels in LP
        ii. bp_init : the initial value of messages in BP
        iii. sp_score : the score associated to short paths
        '''
        
        node.set_observed(False)
        node.set_state(0.)
        node.sp.score = sp_score
        node.visited = False
        self.set_bp_lp_messages(node, lp_init = lp_init, bp_init=bp_init)
        
        return

    def reset_all_nodes(self):
        '''
        A method to reset all sets of nodes in the network.
        It erases all info about perturbed/unperturbed and observed/unobserved nodes
        as well as resetting all values related to inference.
        '''
        
        ## erase info about observed/perturbed nodes
        del self.observed[:]
        del self.perturbed[:]
        ## reset node properties
        for node in self.nodes():
            self.reset_node(node)

        return
    

    def pickled_net_loader (self, file_name, load_pert = True):
        '''
        A method to load a picled network.
        Indeed, nets can be pickled by embedding all info about nodes.
        Mandatory arguments
        i. file_name : the name of the pickle file
        Keyword arguments
        i. load_pert : if true, load the set of perturbed nodes,
                    if false, just load the network.
        '''
        
        ## open the pickle file and load stuff
        obj=open(file_name,"r")
        nodes = pickle.load(obj)
        edges = pickle.load(obj)
        observed = pickle.load(obj)
        obj.close()

        ## add nodes and edges to the current net
        bare_nodes = [inferenceNode(tag=i, state= nodes[i], observed = (i in observed) ) for i in sorted(nodes.keys())]
        for i,j in edges:
            self.add_edge(bare_nodes[i],bare_nodes[j])
        ## del stuff in memory
        del nodes
        del edges
        del observed
        
        ## set some net props
        self.name = '_'.join( file_name.split('/')[-1].split('.')[:-1])
        self.observed = []
        self.perturbed = []
        
        ## load info about perturbed nodes
        if load_pert:
            for node in self.nodes():
                if node.observed:
                    self.observed.append(node)
                if node.state == 1:
                    self.perturbed.append(node)
                self.set_bp_lp_messages(node)

        ## otherwise reset info
        else:
            self.reset_nodes()
        
        self.unobserved = list( set(self.nodes()) - set(self.observed) )
        self.tag2node = {x.tag: x for x in self.nodes()}

        return
    
    def text_file_net_loader(self, file_name):

        '''
        A method that reads the edges from a text file and retains the giant component.
        Mandatory arguments
        i. file_name : the path to the net file
        '''
        
        self.name = '_'.join( file_name.split('/')[-1].split('.')[:-1])
        ## load the graph from file
        g_tmp = nx.read_edgelist(file_name, data = False)
        if not nx.is_connected(g_tmp):
            ## retain the giant component
            g_tmp_subgraphs = list(nx.connected_component_subgraphs(g_tmp))
            g_tmp_subgraphs.sort(key=lambda x: x.size(), reverse=True)
            g_tmp = g_tmp_subgraphs[0]
                
        ## create a net of inferenceNodes
        ## associating names
        self.tag2node = {}
        bare_nodes = []
        for x,y in g_tmp.edges():
            if x not in self.tag2node:
                bare_nodes.append(inferenceNode(tag=x) )
                self.tag2node[x] = bare_nodes[-1]
            if y not in self.tag2node:
                bare_nodes.append(inferenceNode(tag=y) )
                self.tag2node[y] = bare_nodes[-1]

            n1, n2 = self.tag2node[x], self.tag2node[y]
            self.add_edge(n1,n2)
        
        del g_tmp

        return
    
    def h_sapiens_loader (self, net_file, experiments_file):
        '''
        A method to load files with experimental results on human metabolism.
        These files have a precise standard.
        Mandatory arguments:
        i. net_file : the name of the file containing the network
        ii. experimens_file : the path to the file containing the experimental results
        '''
        ## load the net
        self.text_file_net_loader(net_file)
        
        ## load experimental data
        self.experimentally_observed = set([])
        experimental_data = open(experiments_file).readlines()
        
        ix = 1
        for line in experimental_data:
            ## get the perturbed observed nodes (either over/under expressed)
            if line.split()[0]=='Under' or line.split()[0]=='Over':
                known = line.replace('--- ','').split()
                ## set the state of nodes in the net accordingly
                for metab in known[1:]:
                    try:
                        node = self.tag2node[metab]
                        node.state = 1
                        self.experimentally_observed.add(node)
                        ix += 1

                    except KeyError:
                        ## some metabolites are not in the
                        ## reconstruction
                        pass
            ## get the unerturbed observed nodes
            elif line.split()[0]=='Equal':
                known = line.replace('--- ','').split()
                for metab in known[1:]:
                    try:
                        node = self.tag2node[metab]
                        node.state = 0
                        self.experimentally_observed.add(node)
                        ix += 1
                    except KeyError:
                        ## some metabolites are not in the
                        ## reconstruction
                        pass
        return

    def prune_funnels (self, eta):
        '''
        A method to assign state variables to chains of nodes ending into
        an observed perturbed node. This exploits the fact that if an observed
        perturbed node is at the end of a chain of degree-2 nodes, the perturbation
        has nowere else to pass through.
        Mandatory arguments
        i. eta : the exposure model parameter
        Returns
        i. pruned_set : the final (possibly reduced) set of nodes BP will be applied onto
        ii. pert_neighs : the set of nodes neighboring a surely  perturbed node
        '''
        dead_ends = set([])
        pert_neighs = set([])
        all_ok = False
        
        ## iteratively look for unobserved nodes
        ## of degree == 1, connected to nodes whose
        ## unperturbed state is known
        while not all_ok:
            all_ok = True
            ## get a set of possible candidates
            ## i.e. a set of unobserved nodes
            ## whose state is not already inferred
            pruned_set = set(self.unobserved) - dead_ends - pert_neighs
            ## for each node in this candidate set
            for node in pruned_set:
                ## check whether there's some neighbor belonging either
                ## to the perturbed observed set or those nodes identified
                ## as surely perturbed in previous iterations
                pert_obs_neighs = (set( self.neighbors(node) ) & set( self.perturbed ) & set( self.observed) ) | ( set( self.neighbors(node) ) & dead_ends )
                psi_value = eta
                ## if some neighbor is in those sets
                if len(pert_obs_neighs) != 0:
                    ## for each of these nodes
                    for neigh in pert_obs_neighs:
                        ## if they have degree == 1
                        if len(self.neighbors(neigh)) == 1:
                            ## then the perturbation must have surely passed
                            ## through the node. Add the node to the dead_end set &
                            ## set all_ok to False to iterate again
                            all_ok = False
                            psi_value = 1.
                            dead_ends.add(node)
                    ## otherwise add the node to the set of dead_ends
                    ## to keep track it's a neigbor to some perturbed observed
                    if node not in dead_ends:
                        pert_neighs.add(node)
                
                    ## set the messages to a value related
                    ## to what observed
                    node.bp.set_messages (self.neighbors(node), psi_value=1.-psi_value, assign_neighbors = False)
                    node.bp.psi = psi_value
    
        ## get the final list of nodes
        ## where to apply BP
        pruned_set = set(self.unobserved) - dead_ends - pert_neighs
        return list( pruned_set), list( pert_neighs)



    def message_passing(self, eta, prune_branches=True):
        '''
        A method to implement an iteration of message passing.
        Mandatory arguments
        i. eta : the parameter of the exposure model
        Keyword arguments
        i. prune_branches : if True, look for nodes neighboring surely perturbed
                            nodes, fix their marginal and remove them from the
                            list of nodes where BP will be applied
        Returns
        error : the variation between previous and current estimate of the messages
        '''
        ## look for nodes that can be removed from the inference set
        if prune_branches:
            bp_set = self.pruned_nodes
        ## otherwise include all unobserved nodes
        else:
            bp_set = self.unobserved

        ## compute new messages
        for node in bp_set:
            node.compute_bp_messages(eta)

        ## compute the variation
        ## and update messages
        error = 0.
        for node in bp_set:
            error += node.update_bp_messages(eta)
        return error


    def full_bp (self, tol = 1.e-3, max_iter=100, prune_branches=True):
        '''
        A method that fully implements BP until convergence/up to a maximum
        number of iterations.
        Keyword arguments
        i. tol : tolerance parameter to check for convergence
        ii. max_iter : maximum number of iterations to perform
        iii. prune_branches : if True, look for nodes neighboring surely perturbed
                            nodes, fix their marginal and remove them from the
                            list of nodes where BP will be applied
        Returns
        eta :  the estimate of the exposure model parameter
        '''

        ## initialise parameters
        float_N = float( len( self.nodes()))
        error = 1.
        iter = 0
        eta =  float_N/ float( sum( self.degree().values()))
        eta += 0.005
        ## look for nodes that can be removed from the inference set
        if prune_branches:
            self.pruned_nodes, self.pert_neighbors = self.prune_funnels (eta)
            
        ## iterate message passing as in the SI of the paper
        while error > tol and iter < max_iter:

            error = self.message_passing(eta, prune_branches=prune_branches)
            
            ## compute the expected number of
            ## perturbed/unperturbed nodes
            n11 = 0.
            n01 = 0.
            n00 = 0.
            n10 = 0.
            ## by computing the marginal of each node
            for node in self.nodes():
                psi = 1.
                for neigh in self.neighbors(node):
                    psi *= neigh.bp.messages[node]
                n01 += (1. - node.bp.psi)*(1.-psi)
                n11 += node.bp.psi*(1. - psi)
                n00 += (1. - node.bp.psi)*psi
                n10 += node.bp.psi*psi           

            ## update eta (Eq. 12 of the paper)
            eta = n11 /(n11 + n01)
            for node in self.pert_neighbors:
                ## update messages for those nodes that are not
                ## susceptible to BP
                node.bp.set_messages (self.neighbors(node), psi_value=1.-eta, assign_neighbors = False)
                node.bp.psi = eta

            iter += 1
            error /= float_N

        return eta

    def label_propagation (self, tol = 1.e-3, max_iter = 100):
        ''' 
        A method for label propagation implementation.
        It iterates equation until convergence and up
        to a maximum number of iterations.
        
        Keyword arguments
        i. tol : tolerance parameter to check for convergence
        ii. max_iter : maximum number of iterations to perform
        '''
        
        ## initialise parameters
        error = 1.
        iter = 0
        ## loop until convergence/max iter is reached
        while error > tol and iter < max_iter:
            for node in self.unobserved:
                
                l0 = 0.
                l1 = 0.
                ## compute labels
                for neigh in self.neighbors(node):
                    l0 += (1.-neigh.lp.label)/neigh.lp.norm
                    l1 += neigh.lp.label/neigh.lp.norm
                    
                node.lp.label_new = l1/( l0 + l1)

            error = 0.
            ## update labels
            for node in self.unobserved:
                error += math.fabs(node.lp.label - node.lp.label_new)
                node.lp.label = node.lp.label_new

        return
    
    def get_viable_subgraph (self, method, EPSILON=1.e-3):
        '''
        A method to extract a graph were all observed unperturbed
        nodes are removed. This method is useful to reduce the graph to
        a minimal useful set, since observed unperturbed nodes are
        unaffected by the unkown perturbation. It is used by default
        to make inference via shortest paths.
        
        Mandatory arguments
        i. method : a method for edge weight assignation.
                    Either 'bp', 'lp' or 'sp'.
        Keyword arguments
        i. EPSILON : the square root of the possible minimal edge weight
        Returns
        i. H : a networkx graph
        '''
        
        METHODS_LIST = ['bp','lp','sp']
        ## raise error for unknown methods
        if not (method in METHODS_LIST):
            message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )

            raise TypeError(message)
        ## get all observed unperturbed nodes
        to_remove = list( set(self.observed) - set(self.perturbed) )
        ## initialise a networkx graph
        H = nx.Graph()
    
        for x,y in self.edges():
            ## add an edge to the graph H ony if neither end is
            ## an unperturbed observed node
            if x not in to_remove and y not in to_remove:
                ## assign the weight of the graph according
                ## to the inference method
                if method == 'bp':
                    weight= ((1.- x.bp.psi)+EPSILON)*((1.- y.bp.psi)+EPSILON)
                    H.add_edge(x,y, weight= weight)
                
                elif method == 'lp':
                    weight= ((1.- x.lp.label)+EPSILON)*((1.- y.lp.label)+EPSILON)
                    H.add_edge(x,y, weight= weight)
                
                elif method == 'sp':
                    H.add_edge(x,y)
        ## delete the array
        del to_remove
        ## add possible unobserved nodes that were left out
        for extra_node in self.unobserved:
            if extra_node not in H.nodes():
                H.add_node(extra_node)

        return H

        
    def shortest_paths (self, Hsapiens=False):
        '''
        A method to implement inference via shortest paths.
        This seeks all shortest paths among pair of observed
        perturbed nodes and it associates, to each unobserved
        node, a score proportional to the number of paths
        passing through it.
        Keyword arguments
        i. Hsapiens : whether the network is the reconstruction
                    of the human metabolism.
        '''
        import collections
        ## get a graph whose observed unperturbed nodes
        ## are removed
        H = self.get_viable_subgraph('sp')
        pert_path = set([])
        bifurcation_nodes = collections.defaultdict(int)
        bifurcation_paths = []

        ## initialise scores
        for node in self.unobserved:
            node.sp.score = 0.
        
        ix = 0
        ## get the set of observed perturbed nodes
        pert_obs_sp = list( set(self.perturbed) & set(self.observed) )

        ## loop over pairs of observed
        ## perturbed paths
        for n1 in pert_obs_sp[:-1]:
            for n2 in pert_obs_sp[ix+1:]:
                
                try:
                    ## get all shortest paths between these 2 nodes
                    paths=list( nx.all_shortest_paths (H,n1,n2) )
                    ## compute a normalisation proportional
                    ## to the number of paths between the nodes
                    norm = float(len(paths))
                    ## for each path
                    for path in paths:
                        ## for each node in the path
                        for node in path:
                            ## add a score inv. proportional to
                            ## the number of shortest paths
                            node.sp.score += 1./norm
                            
                            ## THE FOLLOWING CODE IS USED
                            ## TO INFER ACTUAL PERTURBATION PATHS,
                            ## NOT TO COMPUTE NODE SCORES
                            if norm != 1.:
                                ## flag that the node participates in a
                                ## path for a pair with multiple paths
                                bifurcation_nodes[node] += 1
                                
                            else:
                                ## otherwise add it to the set
                                ## perturbation path
                                pert_path.add(node)
                            
                    if norm != 1.:
                        bifurcation_paths.append(paths)
                ## ignore cases where no shortest
                ## path can be found.
                except nx.exception.NetworkXNoPath:
                    continue
            ix += 1
            
        ## THE FOLLOWING CODE IS USED
        ## TO INFER ACTUAL PERTURBATION PATHS,
        ## NOT TO COMPUTE NODE SCORES
        ## add to the inferred perturbation
        ## the nodes participating in most of
        ## the multiple paths
        for bifurcation in bifurcation_paths:
            scores = []
            for a_path in bifurcation:
                scores.append(0.)
                for node in a_path:

                    scores[-1] += bifurcation_nodes[node]
                
                        
            best_path_index = scores.index(max(scores))
            for node in bifurcation[best_path_index]:
                pert_path.add(node)

        for node in pert_path:
            node.sp.inferred = True
        del pert_obs_sp
        
        return


    def auc (self, method='all'):
        '''
        A method to compute the AUC of the possible
        inference methods implemented.
        The AUC is computed by computing the fraction of true
        perturbed unobserved ranked above true unperturbed unobserved.
        Keyword arguments
        i. method : The method for which the AUC should be
                    evaluated. Either 'bp','lp','sp', 'all'.
        Returns
        i.a a specific auc if method != 'all'
        i.b all possible auc if method == 'all'
        '''
        
        ## raise an error for unknown methods
        METHODS_LIST = ['bp','lp','sp', 'all']

        if method.lower() not in METHODS_LIST:
            message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )
            raise TypeError(message)


        ## perturbed unobserved nodes
        pert_unobs = set(self.unobserved) & set(self.perturbed)
        ## unperturbed unobserved nodes
        unpert_unobs = set(self.unobserved) - pert_unobs
        ## normalisation factor
        norm = len(pert_unobs) * len(unpert_unobs)
        norm = float(norm)
        ## initialise parameters
        auc_bp = 0.
        auc_lp = 0.
        auc_sp = 0.

        ## for each method, check how many
        ## true perturbed have an higher score
        ## than true unerturbed
        for pert in pert_unobs:
            for unpert in unpert_unobs:
            
                if pert.bp.psi > unpert.bp.psi:
                    auc_bp += 1.
                elif pert.bp.psi == unpert.bp.psi:
                    auc_bp += 0.5


                if pert.lp.label > unpert.lp.label:
                    auc_lp += 1.
                elif pert.lp.label == unpert.lp.label:
                    auc_lp += 0.5

                if pert.sp.score > unpert.sp.score:
                    auc_sp += 1.
                elif pert.sp.score == unpert.sp.score:
                    auc_sp += 0.5
        ## normalise the score
        auc_bp /= norm
        auc_lp /= norm
        auc_sp /= norm
        ## return the corresponding auc
        if method.lower() == 'bp':
            return auc_bp

        elif method.lower() == 'lp':
            return auc_lp

        elif method.lower() == 'sp':
            return auc_sp

        else:

            return auc_bp, auc_lp, auc_sp

    def recall_precision(self, method='all'):
        '''
        A method to compute the recall and precision
        of the possible inference protocols.
        Recall & precision are an alternative metric to AUC,
        being defined as:
        recall = true positive / (true positive + false negative)
        precision = true positive / (true positive + false positive)
        Keyword arguments
        i. method : the inference method for which recall/precision
                    will be computed
        Returns
        i.a recall & precision for a specific inference if method != 'all'
        i.b all possible recall & precision if method == 'all'
        '''
        ## raise an error if method is unknown
        METHODS_LIST = ['bp','lp','sp', 'all']
        
        if method.lower() not in METHODS_LIST:
            message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )
            raise TypeError(message)

        ## get possible sets
        perturbed_and_observed = set(self.observed) & set(self.perturbed)
        perturbed_unobserved =  set(self.perturbed) - set(self.observed)
        
        ## set a threshold for deciding what is inferred as positive
        expected_perturbed_frac = math.ceil( float( len(perturbed_and_observed) ) /float( len (self.observed) ) *float( len (self.unobserved) ) )
        expected_perturbed_frac = int(expected_perturbed_frac)
        
        

        recall = {}
        precision = {}
        nodes = self.unobserved
        
        ## compute recall and precision for various inference methods
        if method in ['bp', 'all']:
            nodes.sort(key = lambda x : x.bp.psi, reverse = True)
            true_positive = set( nodes[:expected_perturbed_frac] ) & perturbed_unobserved
            recall['bp'] = float(len( true_positive)) / float(len( perturbed_unobserved))
            precision['bp'] = float(len( true_positive) ) / float( expected_perturbed_frac)

        if method in ['lp', 'all']:
            nodes.sort(key = lambda x : x.lp.label, reverse = True)
            true_positive = set( nodes[:expected_perturbed_frac] ) & perturbed_unobserved
            recall['lp'] = float( len( true_positive) ) / float( len( perturbed_unobserved) )
            precision['lp'] = float( len( true_positive) ) / float( expected_perturbed_frac)


        if method in ['sp', 'all']:
            nodes.sort(key = lambda x : x.sp.score, reverse = True)
            true_positive = set( nodes[:expected_perturbed_frac] ) & perturbed_unobserved
            recall['sp'] = float( len( true_positive) ) / float( len( perturbed_unobserved))
            precision['sp'] = float( len( true_positive) ) / float( expected_perturbed_frac)

        ## return results
        if method == 'all':
            return recall, precision

        else:
            return recall[method], precision[method]
            
    def reconstruct_paths (self, method, EPSILON = 1.e-3, verbose = False):
        '''
        Infer the actual paths, given the computed scores of the unobserved nodes.
        '''
        
        if method == 'bp':

            ranked_nodes = sorted( self.unobserved , key = lambda x: x.bp.psi, reverse = True)
            H = nx.Graph()
            H.add_nodes_from(set( self.perturbed ) & set( self.observed ))

            for node in H.nodes():
                for neigh in self.neighbors(node):
                    if neigh in H.nodes():
                        H.add_edge(node, neigh)

            i = 0
            while not nx.is_connected(H):
                next_node = ranked_nodes[i]
                H.add_node(next_node)
                for neigh in self.neighbors(next_node):
                    if neigh in H.nodes():
                        H.add_edge(next_node, neigh)
                i += 1


            score1 = float( len( set(H.nodes() ) & set( self.perturbed ) ) ) / float( len( set(H.nodes() ) | set( self.perturbed ) ) )
            sc1 = set(H.nodes())
            sc2 = set(self.perturbed) & set(self.unobserved)
            scn = sc1 & sc2
            sc3 = set(H.nodes()) - (set(self.perturbed) & set(self.observed) )
            scd = sc3 | sc2
            score2 = float(len(scn))/float(len(scd))
            perturbed_set = set(self.perturbed) & set(self.unobserved)
            pert_path = set( H.nodes() )
            recall_n = float( len( ( perturbed_set & pert_path ) ) )
            recall_d = float(len( perturbed_set) )
            recall = recall_n / recall_d
            precision_n = recall_n
            precision_d = float(len( pert_path & set(self.unobserved) ) )
            
            for node in self.nodes():
                node.bp.inferred = False
            for node in pert_path & set(self.unobserved):
                node.bp.inferred = True

            try:
                precision = precision_n / precision_d
            except ZeroDivisionError:
                precision = 0.

            if verbose:
                print "\t (BP) # of unobserved nodes in H %d, unobs pert. size %d" %(len(sc1 & set(self.unobserved) ), len(sc2) )
            return score1, score2, recall, precision

        elif method == 'sp-score':

            ranked_nodes = sorted( self.unobserved , key = lambda x: x.sp.score, reverse = True)
            H = nx.Graph()
            for node in set( self.perturbed ) & set( self.observed ):
                H.add_node(node)
            for node in H.nodes():
                for neigh in self.neighbors(node):
                    if neigh in H.nodes():
                        H.add_edge(node, neigh)

            i = 0
            while not nx.is_connected(H):
                next_node = ranked_nodes[i]
                H.add_node(next_node)
                for neigh in self.neighbors(next_node):
                    if neigh in H.nodes():
                        H.add_edge(next_node, neigh)
                i += 1


            score1 = float( len( set(H.nodes() ) & set( self.perturbed ) ) ) / float( len( set(H.nodes() ) | set( self.perturbed ) ) )
            sc1 = set(H.nodes())
            sc2 = set(self.perturbed) & set(self.unobserved)
            scn = sc1 & sc2
            sc3 = set(H.nodes()) - (set(self.perturbed) & set(self.observed) )
            scd = sc3 | sc2
            score2 = float(len(scn))/float(len(scd))
            perturbed_set = set(self.perturbed) & set(self.unobserved)
            pert_path = set( H.nodes() )
            recall_n = float( len( ( perturbed_set & pert_path ) ) )
            recall_d = float(len( perturbed_set) )
            recall = recall_n / recall_d
            precision_n = recall_n
            precision_d = float(len( pert_path & set(self.unobserved) ) )

            try:
                precision = precision_n / precision_d
            except ZeroDivisionError:
                precision = 0.
            for node in self.nodes():
                node.sp.inferred = False
            for node in pert_path & set(self.unobserved):
                node.sp.inferred = True

            if verbose:
                print "\t (SP-score) # of unobserved nodes in H %d, unobs pert. size %d" %(len(sc1 & set(self.unobserved) ), len(sc2) )
            return score1, score2, recall, precision


        elif method == 'sp':

            score1 = self.infer_paths('sp')
            pert_path = set([])

            for node in self.unobserved:
                if node.sp.inferred:
                    pert_path.add(node)


            perturbed_set = set(self.perturbed) & set(self.unobserved)
            norm = len( perturbed_set | pert_path )
            score2 = float( len( ( perturbed_set & pert_path ) ) ) / float(norm)
            recall_n = float( len( ( perturbed_set & pert_path ) ) )
            recall_d = float(len( perturbed_set) )
            recall = recall_n / recall_d
            precision_n = recall_n
            precision_d = float(len( pert_path & set( self.unobserved) ) )

            try:
                precision = precision_n / precision_d

            except ZeroDivisionError:
                precision = 0.

            for node in self.nodes():
                node.sp.inferred = False


            for node in pert_path & set(self.unobserved):
                node.sp.inferred = True


            if verbose:
                print "\t (SP) # of inferred unobserved nodes %d, unobs pert. size %d" %(len(pert_path), len(perturbed_set) )

            return score1, score2, recall, precision
                    

    def si_pert(self, p, root, triangular = False, k = None, homog = None):

        ''' A recursive function to create a perturbation across the network '''

        ## if root node is not given, choose it at random
        if not root:
            root = random.choice(self.nodes())

        ## set the status of root node to visited & perturbed

        root.visited = True
        root.state = 1
        p_prime = p
        if not p_prime:
            if triangular:
                lmbda = 1./k+0.05
                p_prime = random.triangular(lmbda*0.5,min(1.,lmbda*1.5),lmbda)
            elif homog:
                p_min, p_max = homog
                if p_min == p_max:
                    p_prime = p_min

                else:
                    p_prime = random.uniform(p_min, p_max)

            else:
                p_prime = random.random()


        self.perturbed.append(root)

        ## infect non visited neighbors of root node with probability p

        for neigh in self.neighbors(root):

            if not neigh.visited and random.random() < p_prime:

                ## by calling recursively the function
                self.si_pert(p, neigh, triangular = triangular, k = k, homog = homog)

            elif not neigh.visited:
                
                neigh.visited = True
        return

    def create_perturbation (self, p, ob, triangular=False, homog = None):


        ''' A method that effectively creates the perturbation given the network and returns a list of perturbed nodes '''

        self.reset_all_nodes()
        obs_length = int( ob * float( len(self.nodes() ) ) )

        if triangular :
            import numpy as np 
            k = float ( np.mean(self.degree().values()) )

        else:
            k = None


        while len(set(self.perturbed) & set(self.observed)) == 0 or len(set(self.perturbed) - set(self.observed)) == 0 or len(set(self.observed) - set(self.perturbed))==0 or len(self.observed)==0:

            self.reset_all_nodes()

            ## call the perturbating routine
            self.si_pert(p, None, triangular=triangular, k = k, homog = homog)
            
            ## get an observed set
            self.observed = random.sample(self.nodes(),obs_length)

        self.unobserved = list( set(self.nodes()) - set(self.observed) )
        
        for node in self.observed:
        
            node.observed = True
        
        for node in self.nodes():

            self.set_bp_lp_messages(node)

        return

    def hibiscus_setup (self, fraction):
        '''
        A method to initialise the network according to the experimental results
        on human metabolism
        '''
        del self.perturbed[:]
        del self.observed[:]
        
        obs_length = int( fraction * float( len(self.experimentally_observed ) ) )
        for node in self.experimentally_observed:
            if node.state == 1:
                self.perturbed.append(node)

        while len(set(self.perturbed) & set(self.observed)) == 0 or (len(set(self.perturbed) - set(self.observed)) == 0 and fraction != 1.)  or len(set(self.observed) - set(self.perturbed))==0 or len(self.observed)==0:
            del self.observed[:]
            
            self.observed = random.sample(self.experimentally_observed,obs_length)
        
        self.unobserved = list( set(self.nodes()) - set(self.observed) )
        for node in self.observed:
                node.observed = True
                                    
        for node in self.nodes():
                self.set_bp_lp_messages(node)
                    
        return
                    
                


    def generate (self, N, k_av, graph_type = 'erdos_renyi'):
        '''
        Generate the network according to a set of predefined models
        '''
    
        ALLOWED_TYPES = ['erdos_renyi', 'watts_strogatz', 'barabasi_albert']
        if graph_type.lower() not in ALLOWED_TYPES:
            message = 'Unknown graph type %s.\n Graph type should be one of %s' %(method, ', '.join(ALLOWED_TYPES) )
    
            raise TypeError(message)

        elif graph_type == 'erdos_renyi':
            self.name = 'Erdos-Renyi'
            p = k_av/float(N -1)
            g_tmp = nx.erdos_renyi_graph(N, p)
            while not nx.is_connected(g_tmp):
                g_tmp = nx.erdos_renyi_graph(N, p)

        elif graph_type == 'watts_strogatz':
            self.name = 'Watts-Strogatz'
            g_tmp = nx.connected_watts_strogatz_graph(N, int(k_av), 0.3, tries=500)

        elif graph_type == 'barabasi_albert':
            import numpy as np
            self.name = 'Barabasi-Albert'
            M = int( float(N) * k_av * 0.5 )
            coeffs = np.array([-1., float(N), -float(M)])
            mPrime = np.roots(coeffs)
            
            if np.min(mPrime)>0.:
                m = np.min(mPrime)
            else:
                m = np.max(mPrime)
            m = int(np.round(m))

            g_tmp = nx.barabasi_albert_graph(N, m)
            while not nx.is_connected(g_tmp):
                g_tmp = nx.barabasi_albert_graph(N, m)

        bare_nodes = [ inferenceNode(tag=i) for i in range(N) ]
        for x,y in g_tmp.edges():
            self.add_edge( bare_nodes[x], bare_nodes[y] )

        del g_tmp

        return

    def compute_layout(self, alpha = 0.48, scale = 1.):
        '''
        compute the layout of the network in order to plot it
        '''
        k = 1./( float( len( self.nodes() ) )** alpha)
        self.layout = nx.spring_layout(self, k=k, scale = scale)

        return



if __name__ == '__main__':
    
    import sys
    
    G = inferenceGraph()
    
    G.h_sapiens_loader()
    
    G.hibiscus_setup(0.5)
    
    nodefile = open('Hsapiens_nodes.dat','w')
    for node in G.nodes():
        
        extra = ''

        if node in G.perturbed:

            extra = 'Perturbed'


        elif node in G.observed:

            extra = 'Unperturbed'

        print >>nodefile, "%s %s" %(node.tag, extra)


    nodefile.close()

    edgefile = open('Hsapiens_edges.dat','w')


    for x,y in G.edges():

        print >>edgefile,"%s %s" %(x.tag, y.tag)

    edgefile.close()


        
    exit(0)

    print G
    
    G.full_bp()
    
    G.label_propagation()
    
    ##    G.shortest_paths()
    
    print G.auc()
    
    exit(0)
    
    G.text_file_net_loader(sys.argv[1])
    
    ##G.generate (1000, 4., 'barabasi_albert')

    G.create_perturbation(0.49, 0.5)





    print G.recall_precision()

    print G.infer_paths ('bp')

    print G.infer_paths ('lp')

    print G.infer_paths ('sp')



