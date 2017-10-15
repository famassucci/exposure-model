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
