import math

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
            ## according to the Eq. 4 of the paper
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
        ## compute the marginal psi (Eq. 5 of the paper)
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
