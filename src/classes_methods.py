import math, random
import networkx as nx
import cPickle as pickle

class BPProps ():

    '''
    A class to deal with Belief-Propagation messages.
    It handles all messages, the update of these and the actual marginal
    '''

    def __init__ (self, P1 = 0.5, neighbors = None):
        '''
        A method to initialise messages and distributions.
        Keyword arguments
        i. P1 : the initial value of the marginal
        ii. neighbors : a list of neighboring nodes to propagate messages
        '''

        ## initialise marginal and messages
        ## messages are dealt by with dictionaries
        ## each neighbor is a dict key
        self.psi = P1
        self.messages = {}
        self.messages_new = {}
        self.cavity_neighbors = {}
        self.inferred = False
    
        if neighbors:
            ## for each neighbor provided
            for neigh in neighbors:
                ## initialise the messages to .5
                ## get the cavity neighbors from which
                ## messages are received in absence of neigh
                self.messages[neigh] = 0.5
                self.messages_new[neigh] = 0.5
                self.cavity_neighbors[neigh] = list(
                    set(neighbors) - set([neigh])
                )

        return

    def set_messages (self, neighbors, psi_value=0.5, assign_neighbors = True):
        '''
        A method to set messages to a certain value.
        Messages are thus created from scratch and stored in a dict.
        Mandatory arguments
        i. neighbors : a list of neighbors of the current node
        Keyword arguments
        i. psi_value : the value assigned to the messages
        ii. assign_neighbors : if True also assign psi_value to the cavity messages
        '''
        
        ## initialise messages
        self.messages = {}
        self.messages_new = {}

        ## for each neighbor
        for neigh in neighbors:
            ## fix the message (and its updated value) to psi_value
            self.messages[neigh] = psi_value
            self.messages_new[neigh] = psi_value
            if assign_neighbors:
                self.cavity_neighbors[neigh] = list(
                    set(neighbors) - set([neigh])
                )
        return




  
class LPProps ():
    '''
    A class to deal with Label-propagation.
    It is used to handle the propagation of labels
    in the network.
    '''
    
    def __init__(self, label=0.5, neighbors=None):
        '''
        A method to initialise labels.
        Keyword arguments
        i. label : the initial label value
        ii. neighbors : either a list of neighbors or the number of neighbors.
                        this is used to normalise the label value
        '''
        
        ## set the label value
        self.label = label
        self.label_new = label
        self.inferred = False

        ## normalise the label if neighbors keyword is provided
        if neighbors:
            try:
                self.norm = float(neighbors)
            except TypeError:
                self.norm = float(len(neighbors))
        else:
            self.norm = 1.
        return

class SPProps ():
    '''
    A class to deal with short paths scoring.
    It is used to deal with shortest path inference.
    '''
    def __init__(self, score = 0., in_path = False):
        '''
        A method to initialise the score.
        Keyword arguments
        i. score : an arbitrary value for the perturbation score
        ii. in_path :
        '''
        self.score = score
        self.inferred = in_path

        return
            

class inferenceNode ():
    '''
    A class for the nodes in the network.
    It allows to:
    1. generate perturbations in the network
    2. infer perturbations by:
        a. belief propagation (BP)
        b. label propagation (LP)
        c. shortest paths (SP)
    using the dedicated classes.
    '''

    def __init__ (self, state=0, tag = None, observed = False, lp_init = 0.5, bp_init=0.5):
        '''
        A method to initialise all properties of the node.
        Keyword arguments:
        i. state : the state (perturbed = 1 / unperturbed = 0) of the node
        ii. tag : a name for the node
        iii. observed : wether the node is observed (True) or not (False)
        iv. lp_init : initial value for the labels to be used in label propagation
        v. bp_init : initial value for the messages to be used in belief propagation
        '''
        ## initialise all properties
        self.tag = tag
        self.observed = observed
        self.state = state
        self.visited = False
        self.bp = BPProps(P1 = bp_init)
        self.lp = LPProps(label = lp_init)
        self.sp = SPProps()

        return

    def __repr__ (self):
        '''
        A method for representing nodes when printing.
        '''
        ## build a string which is the composite
        ## of the properties of the node
        line = []
        line.append('id: ')
        line[-1] += str(self.tag)
        line.append('obs: ')
        line[-1] += str(self.observed)
        line.append('state: ')
        line[-1] += str(self.state)

        return '; '.join(line)

    def set_state(self, state):
        '''
        A method to set the state (perturbed/unperturbed) of the nodes.
        Mandatory argument:
        i. state : the state (perturbed = 1/ unperturbed = 0) of the node
        '''
        ## set the state of the node either to 0 or 1
        if state != 0. and state != 1.:
            raise ValueError()
        self.state = state

        return

    def set_observed(self, observed):
        '''
        A method to set whether the node is observed or not.
        Mandatory argument:
        i. observed : whether the node is observed (True) or not (False)
        '''
        ## if the node is observed
        if observed:
            ## put the property observed to True
            self.observed = True
        else:
            ## otherwise put it to false
            self.observed = False

        return

    def compute_bp_messages(self, eta):
    
        '''
        A method to compute the outgoing messages with BP.
        Mandatory arguments:
        eta : the exposure model parameter
        '''
        
        ## for each neighbor
        for neigh in self.bp.messages_new:
            psi = 1.
            ## the message propagated is related to the product
            ## of all incoming messages when the neighbor is removed
            ## so loop over the 'cavity neighbors' i.e. all neighbors but neigh
            for c_neigh in self.bp.cavity_neighbors[neigh]:
                psi *= c_neigh.bp.messages[self]
            ## compute the new message propagated to neigh
            ## according to the formula XX of the paper
            self.bp.messages_new[neigh] = (1.-eta)*(1.-psi)+ psi
    
        ## if there's only one neighbor
        if len(self.bp.messages) == 1:
            ## do a trick and compute the message based
            ## on the message propagated by the neighbor
            neigh = self.bp.messages_new.keys()[0]
            self.bp.messages_new[neigh] = 1. - eta * (1. - neigh.bp.messages[self])

    def update_bp_messages(self, eta):
        '''
        A method to update the messages.
        This basically does the following:
        a. computes the difference among the previous and current estimate;
        b. updates the current estimate of the messages;
        c. computes the estimates for the marginal psi.
        
        The difference among message estimatees is returned to check for convergence.
        Mandatory argument:
        i. eta : the exposure model parameter
        '''
        ## initialise the difference between old and current
        ## estimate of messages to 0 and set the marginal psi to 1.
        error = 0.
        psi = 1.
        
        for neigh in self.bp.messages:
            ## compute the difference between current and old message estimate
            error += math.fabs( self.bp.messages[neigh] - self.bp.messages_new[neigh])
            ## and update messages
            self.bp.messages[neigh] = self.bp.messages_new[neigh]
            ## update the actual
            psi *= neigh.bp.messages_new[self]
        ## compute the marginal psi (formula XX of the paper)
        p0 = (1.-eta)*(1.-psi) + psi
        self.bp.psi = 1. - p0

        return error

    def compute_bp_P (self, eta):
        '''
        A method to compute the marginals psi of each node.
        Mandatory arguments:
        i. eta : the exposure model parameter
        '''
        ## initialise the marginal psi to 1.
        psi = 1.
        for neigh in self.bp.messages:
            ## do the products
            psi *= neigh.bp.messages[self]
        
        ## compute p0/ psi
        p0 = (1.-eta)*(1.-psi) + psi
        self.bp.psi = 1. - p0

        return



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
        
        N = len(self.nodes())
        if N>0:
            k_av = float( sum( self.degree().values() ) ) / float( N )
        else:
            k_av = 0.

        graph_line = 'Inference graph %s of %d nodes, average degree %g' %(self.name, N, k_av)
        p_size = len(self.perturbed)
        obs_size = len(self.observed)
        pert_obs = len(set(self.observed) & set(self.perturbed))
        unpert_obs = len( set(self.observed) - (set(self.observed) & set(self.perturbed))) ##obs_size - pert_obs
        obs_line = '%d perturbed nodes %d observed nodes (of which %d pert and %d unpert)' %(p_size, obs_size, pert_obs, unpert_obs)

        return '; '.join([graph_line, obs_line])
    
    def __str__ (self):
        '''
        A string method to print the class
        '''
        return self.__repr__()
    
    def set_bp_lp_messages(self, node, lp_init = 0.5, bp_init=0.5):
        '''
        Set up messages for BP and LP
        '''

        if node.observed:

            node.bp.psi = node.state
            node.bp.set_messages (self.neighbors(node), psi_value=1.-node.state)
            node.lp.label = node.state
            node.lp.norm = float(len(self.neighbors(node)))

        else:
            
            node.bp.psi = bp_init
            node.bp.set_messages (self.neighbors(node), psi_value=bp_init)
            node.lp.label = lp_init
            node.lp.norm = float(len(self.neighbors(node)))

        return

    def reset_node(self, node, lp_init = 0.5, bp_init=0.5, sp_score = 0.):
        
        '''
        Reset the node properties
        '''
        
        node.set_observed(False)
        node.set_state(0.)
        node.sp.score = sp_score
        node.visited = False
        self.set_bp_lp_messages(node, lp_init = lp_init, bp_init=bp_init)
        
        return

    def reset_all_nodes(self):
        '''
        Reset all sets of nodes in the network and their respective state
        '''
        
        del self.observed[:]
        del self.perturbed[:]
        
        for node in self.nodes():
            self.reset_node(node)

        return
    

    def pickled_net_loader (self, file_name, load_pert = True):
        '''
        Load a picled network
        '''
        
        obj=open(file_name,"r")
    
        nodes = pickle.load(obj)
        edges = pickle.load(obj)
        observed = pickle.load(obj)
    
        obj.close()

        bare_nodes = [inferenceNode(tag=i, state= nodes[i], observed = (i in observed) ) for i in sorted(nodes.keys())]

        for i,j in edges:

            self.add_edge(bare_nodes[i],bare_nodes[j])

        del nodes
        del edges
        del observed
        
        self.name = '_'.join( file_name.split('/')[-1].split('.')[:-1])
        self.observed = []
        self.perturbed = []
        

        if load_pert:

            for node in self.nodes():
                
                if node.observed:
                    self.observed.append(node)
                if node.state == 1:
                    self.perturbed.append(node)
        
                self.set_bp_lp_messages(node)

        else:
            self.reset_nodes()
        
        self.unobserved = list( set(self.nodes()) - set(self.observed) )
        self.tag2node = {x.tag: x for x in self.nodes()}

        return
    
    def text_file_net_loader(self, file_name):

        ''' A function that reads the edges from the text file and creates a nx ready graph '''
       
        edges = []
        num_nodes=-1

        self.name = '_'.join( file_name.split('/')[-1].split('.')[:-1])

        
        g_tmp = nx.Graph()
        
        for line in open(file_name).readlines():

            node1 = line.split()[0]
            node2 = line.split()[1]

            g_tmp.add_edge(node1,node2)
            
        if not nx.is_connected(g_tmp):
        
            g_tmp_subgraphs = list(nx.connected_component_subgraphs(g_tmp))
            
            g_tmp_subgraphs.sort(key=lambda x: x.size(), reverse=True)
            
            g_tmp = g_tmp_subgraphs[0]
                
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
        Load a file with experimental results on human metabolism
        '''
        self.text_file_net_loader(net_file)
        self.experimentally_observed = set([])
        experimental_data = open(experiments_file).readlines()
        
        ix = 1
        for line in experimental_data:

            if line.split()[0]=='Under' or line.split()[0]=='Over':
                known = line.replace('--- ','').split()
                
                for metab in known[1:]:
                    try:
                        node = self.tag2node[metab]
                        node.state = 1
                        self.experimentally_observed.add(node)
                        ix += 1

                    except KeyError:
                        pass

            elif line.split()[0]=='Equal':
                
                known = line.replace('--- ','').split()
                for metab in known[1:]:
                    try:
                        node = self.tag2node[metab]
                        node.state = 0
                        self.experimentally_observed.add(node)
                        ix += 1
                    except KeyError:
                    
                        pass
        return

    def prune_funnels (self, eta):
        '''
        Assign state variables to dead ends connected through an observed node
        '''
        dead_ends = set([])
        pert_neighs = set([])
        all_ok = False
        
        while not all_ok:
            all_ok = True
            pruned_set = set(self.unobserved) - dead_ends - pert_neighs
            
            for node in pruned_set:
                pert_obs_neighs = (set( self.neighbors(node) ) & set( self.perturbed ) & set( self.observed) ) | ( set( self.neighbors(node) ) & dead_ends )
                psi_value = eta
                
                if len(pert_obs_neighs) != 0:
                    for neigh in pert_obs_neighs:
                        if len(self.neighbors(neigh)) == 1:
                            
                            all_ok = False
                            psi_value = 1.
                            dead_ends.add(node)
            
                    if node not in dead_ends:
                        pert_neighs.add(node)
                

                    node.bp.set_messages (self.neighbors(node), psi_value=1.-psi_value, assign_neighbors = False)
                    node.bp.psi = psi_value
    
    
        pruned_set = set(self.unobserved) - dead_ends - pert_neighs
        return list( pruned_set), list( pert_neighs)



    def message_passing(self, eta, prune_branches=True):
        ''' A method to implement the message passing bit of Belief propagation'''
        
        if prune_branches:
            bp_set = self.pruned_nodes

        else:
            bp_set = self.unobserved


        for node in bp_set:
            node.compute_bp_messages(eta)

        error = 0.

        for node in bp_set:
            error += node.update_bp_messages(eta)
        return error


    def full_bp (self, tol = 1.e-3, max_iter=100, prune_branches=True):
        '''
        BP implementation, with update of all parameters
        '''

        float_N = float( len( self.nodes()))
        error = 1.
        iter = 0
        eta =  float_N/ float( sum( self.degree().values()))
        eta += 0.005
        
        if prune_branches:
            self.pruned_nodes, self.pert_neighbors = self.prune_funnels (eta)
            

        while error > tol and iter < max_iter:

            error = self.message_passing(eta, prune_branches=prune_branches)
            n11 = 0.
            n01 = 0.
            n00 = 0.
            n10 = 0.

            for node in self.nodes():
            
                psi = 1.
                for neigh in self.neighbors(node):
                    psi *= neigh.bp.messages[node]
                        

                n01 += (1. - node.bp.psi)*(1.-psi)
                n11 += node.bp.psi*(1. - psi)
                n00 += (1. - node.bp.psi)*psi
                n10 += node.bp.psi*psi           


            eta = n11 /(n11 + n01)
            for node in self.pert_neighbors:
                node.bp.set_messages (self.neighbors(node), psi_value=1.-eta, assign_neighbors = False)
                node.bp.psi = eta

            iter += 1
            error /= float_N

        return eta

    def label_propagation (self, tol = 1.e-3, max_iter = 100):
        ''' 
        Label propagation implementation
        '''
        
        error = 1.
        iter = 0

        while error > tol and iter < max_iter:
            for node in self.unobserved:
                
                l0 = 0.
                l1 = 0.
                for neigh in self.neighbors(node):
                    
                    l0 += (1.-neigh.lp.label)/neigh.lp.norm
                    l1 += neigh.lp.label/neigh.lp.norm
                    
                node.lp.label_new = l1/( l0 + l1)

            error = 0.
            for node in self.unobserved:
                error += math.fabs(node.lp.label - node.lp.label_new)
                node.lp.label = node.lp.label_new

        return
    
    def get_viable_subgraph (self, method, EPSILON=1.e-3):
        '''
        A method to extract a graph were all observed unperturbed nodes are removed.
        '''
        to_remove = list( set(self.observed) - set(self.perturbed) )
        H = nx.Graph()
    
        for x,y in self.edges():
            if x not in to_remove and y not in to_remove:
                if method == 'bp':
                    weight= ((1.- x.bp.psi)+EPSILON)*((1.- y.bp.psi)+EPSILON)
                    H.add_edge(x,y, weight= weight)
                
                elif method == 'lp':
                    weight= ((1.- x.lp.label)+EPSILON)*((1.- y.lp.label)+EPSILON)
                    H.add_edge(x,y, weight= weight)
                
                elif method == 'sp':
                    H.add_edge(x,y)

                else:
                    METHODS_LIST = ['bp','lp','sp']
                    message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )

                    raise TypeError(message)


        del to_remove
        for extra_node in self.unobserved:
            if extra_node not in H.nodes():
                H.add_node(extra_node)

        return H

        
    def shortest_paths (self, Hsapiens=False):
        '''
        Implement inference via shortes paths
        '''
        
        H = self.get_viable_subgraph('sp')
        pert_path = set([])
        bifurcation_nodes = {}
        bifurcation_paths = []

        for node in self.unobserved:
            node.sp.score = 0.
        
        ix = 0
        pert_obs_sp = list( set(self.perturbed) & set(self.observed) )

        for n1 in pert_obs_sp[:-1]:
            for n2 in pert_obs_sp[ix+1:]:
            
                try:
                    paths=list( nx.all_shortest_paths (H,n1,n2) )
                    norm = float(len(paths))

                    for path in paths:
                        for node in path:
                            node.sp.score += 1./norm
                            if norm != 1.:
                                if node not in bifurcation_nodes:
                                    bifurcation_nodes[node] = int(node in pert_path)
                                bifurcation_nodes[node] += 1
                                
                            else:
                                pert_path.add(node)
                            
                    if norm != 1.:
                        bifurcation_paths.append(paths)
                    
                except nx.exception.NetworkXNoPath:
                    continue
            ix += 1
            
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
        compute the AUC of the inference
        '''
        
        METHODS_LIST = ['bp','lp','sp', 'all']

        if method.lower() not in METHODS_LIST:
            message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )
            raise TypeError(message)



        pert_unobs = set(self.unobserved) & set(self.perturbed)
        unpert_unobs = set(self.unobserved) - pert_unobs
        norm = len(pert_unobs) * len(unpert_unobs)


        norm = float(norm)
        auc_bp = 0.
        auc_lp = 0.
        auc_sp = 0.

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

        auc_bp /= norm
        auc_lp /= norm
        auc_sp /= norm

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
        compute the recall and precision of the inference
        '''
        
        METHODS_LIST = ['bp','lp','sp', 'all']
        
        if method.lower() not in METHODS_LIST:
            message = 'Unknown method %s.\n Method should be one of %s' %(method, ', '.join(METHODS_LIST) )
            raise TypeError(message)


        perturbed_and_observed = set(self.observed) & set(self.perturbed)
        expected_perturbed_frac = math.ceil( float( len(perturbed_and_observed) ) /float( len (self.observed) ) *float( len (self.unobserved) ) )
        expected_perturbed_frac = int(expected_perturbed_frac)
        perturbed_unobserved =  set(self.perturbed) - set(self.observed)

        recall = {}
        precision = {}
        nodes = self.unobserved
        
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
        A method to initialise the network according to the exerimental results
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
        Generate the newtork according to a set of predefined models
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



