from sklearn import tree
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import bokeh.plotting as bplt

class tree_plot:
    '''
    class to plot decision tree
    this only works for binary trees right now
    '''
    def __init__(self, clf):
        self.tree = clf.tree_
        self.tree_map = {}
        self.left = self.tree.children_left
        self.right = self.tree.children_right
        self.clf = clf

        self.layers = set()
        self.map_init()
        self.edges = []

        # find relative x values for nodes
        # don't need to find x value for top layer
        for l in sorted(self.layers)[:-1]:
            node_list = filter(lambda n: self.tree_map[n]['y'] == l,
                               self.tree_map)
            for node in node_list:
                self.tree_map[node]['x'] = self.bottom_up(node)
        # sum the relative x values to get the absolute x values
        self.x_abs(0)
        # center parents above their children
        for l in sorted(self.layers):
            node_list = filter(lambda n: self.tree_map[n]['y'] == l,
                               self.tree_map)
            for node in node_list:
                self.center(node)



    def map_init(self, node = 0, y = 0, parent = None):
        '''
        calcualte y values for nodes based on the tree layer
        note: recursive function
        '''
        self.layers.add(y)

        # check for terminal node
        if self.left[node] == -1:
            self.tree_map[node] = {
                'y':y,
                'x':0,
                'parent':parent,
            }
        else:
            self.tree_map[node] = {
                'y':y,
                'x':0,
                'parent':parent,

                'r_child':self.right[node],
                'l_child':self.left[node],
            }

            self.map_init(node = self.right[node], y = y-1, parent = node)
            self.map_init(node = self.left[node], y = y-1, parent = node)


    def bottom_up(self, node):
        '''
        build decision tree from the bottom up
        '''
        # find direction of node from parent
        p = self.tree_map[node]['parent']
        if self.tree_map[p]['l_child'] == node:
            direction = -1
        else:
            direction = 1
        # if node is terminal return 0.5 distance
        if self.left[node] == -1:
            self.tree_map[node]['r_dist'] = 0
            self.tree_map[node]['l_dist'] = 0
            return 0.5 * direction

        l_child = self.tree_map[self.tree_map[node]['l_child']]
        r_child = self.tree_map[self.tree_map[node]['r_child']]

        self.tree_map[node]['r_dist'] = max(
            (r_child['r_dist'] + r_child['x']),
            # remember the left child has a negative x
            (l_child['r_dist'] + l_child['x']))

        self.tree_map[node]['l_dist'] = 0.5 + max(
            (r_child['l_dist'] - r_child['x']),
            # remember the left child has a negative x
            (l_child['l_dist'] - l_child['x']))

        # the x of a left child is 0.5 + the right distance of its right child
        if direction == -1:
            return sum([0.5, self.tree_map[self.right[node]]['x'],
                       r_child['r_dist']]) * -1
       # the x of a right child is 0.5 + the left distance of its left child
        else:
            return sum([-0.5, self.tree_map[self.left[node]]['x'],
                       -l_child['l_dist']]) * -1


    def x_abs(self, node = 0, x = 0):
        '''
        sums the relative x position to find absolute x position
        '''
        x += self.tree_map[node]['x']
        self.tree_map[node]['x_abs'] = x

        if self.left[node] == -1:
            return

        self.x_abs(self.right[node], x)
        self.x_abs(self.left[node], x)


    def center(self, node):
        '''
        center a node between its children
        '''
        # if node is terminal do nothing
        if self.left[node] == -1:
            return
        left = self.tree_map[self.left[node]]['x_abs']
        right = self.tree_map[self.right[node]]['x_abs']
        self.tree_map[node]['x_abs'] = (left + right)/2.0


    def plot(self, feature_names = None, round_n = 2,
             palette = sns.diverging_palette(
                255, 133, l=60, n=15, center="light")):
        '''
        actual plot function
        '''
        if feature_names == None:
            feature_names = range(self.tree.n_features)

        # find appropriate dimensions for the plot
        x_max = max(self.tree_map[n]['x_abs'] for n in self.tree_map)
        x_min = min(self.tree_map[n]['x_abs'] for n in self.tree_map)
        x_size = x_max - x_min
        y_size = max(abs(self.tree_map[n]['y']) for n in self.tree_map)
        plt.figure(figsize = (x_size/1.25, y_size*2))
        # initialize networkx plot
        G = nx.Graph()
        # add edges
        for i,n in enumerate(self.left):
            if n != -1:
                G.add_edge(i,n,weight=1)
        for i,n in enumerate(self.right):
            if n != -1:
                G.add_edge(i,n,weight=1)

        labels = {}
        white_nodes = []
        colored_nodes = []
        node_colors = []
        pal_len = len(palette) - 1
        # node lables
        for i,j in enumerate(self.tree.feature):
            # if node is a decision it should be white
            if j != -2:
                labels[i] = str(feature_names[j]) + '\n'
                labels[i] +='>=' + str(np.round(self.tree.threshold[i],round_n))
                white_nodes.append(i)
            # if node is terminal it should be colored
            # only works for binary decision trees
            else:
                # label = str(np.argmax(self.tree.value[i])) + '\n'
                colored_nodes.append(i)
                color = self.tree.value[i][0][1]/np.sum(self.tree.value[i][0])
                label = str(round(color,2))
                label += '\n' + str(int(np.sum(self.tree.value[i][0])))
                labels[i] = label
                cix = round(color * pal_len,0)
                node_colors.append(palette[int(cix)])
        # build location map for nodes
        pos = {int(i):(self.tree_map[i]['x_abs'], self.tree_map[i]['y'])
               for i in xrange(len(self.tree_map))}

        # draw nodes
        nodes = nx.draw_networkx_nodes(G,pos, nodelist = white_nodes,
                                       node_color = '#FFFFFF',
                                       node_size=2000, node_shape = 'o')
        # remove decision node borders
        nodes.set_edgecolor('#FFFFFF')
        # add colored nodes
        nx.draw_networkx_nodes(G,pos, nodelist = colored_nodes,
                               node_size=1000, node_color = node_colors)
        # draw edges
        nx.draw_networkx_edges(G,pos,alpha=0.5,width=6)
        # add lables
        nx.draw_networkx_labels(G,pos,labels,font_size=12,
                                font_family='sans-serif')
        # nx.draw_networkx_labels(G,pos,font_size=12,
        #                         font_family='sans-serif')
        plt.axis('off')
        plt.xlim(min(pos[x][0] for x in pos)-1, max(pos[x][0] for x in pos)+1)
        plt.show()


    def get_edges(self, node = 0):
        '''
        create the edges for bplot
        '''
        # stopping condition == terminal node
        if self.left[node] == -1:
            return

        # get node location
        start_y = self.tree_map[node]['y']
        start_x = self.tree_map[node]['x_abs']
        # left child location
        l_y = self.tree_map[self.left[node]]['y']
        l_x = self.tree_map[self.left[node]]['x_abs']
        # add edge
        self.edges.append([[start_x, l_x], [start_y, l_y]])
        # right child location
        r_y = self.tree_map[self.right[node]]['y']
        r_x = self.tree_map[self.right[node]]['x_abs']
        # add edge
        self.edges.append([[start_x, r_x], [start_y, r_y]])
        # recurse
        self.get_edges(self.left[node])
        self.get_edges(self.right[node])


    def bplot(self, feature_names = None, round_n = 2, output_file = None,
              size = (800, 600), palette = sns.diverging_palette(255, 133, l=60,
              n=15, center="light")):
        '''
        bokeh plot decision tree
        '''
        # initialize plot
        p = bplt.figure(plot_width=size[0], plot_height=size[1])
        # save output if requested
        if output_file != None:
            bplt.output_file(output_file)

        if feature_names == None:
            feature_names = range(self.tree.n_features)

        # find edges in tree
        self.get_edges()
        # add edges to plot
        for ed in self.edges:
            p.line(ed[0], ed[1], line_width = 2)

        # node labels and coloring
        labels = {}
        node_colors = []
        pal_len = len(palette) - 1
        purity = {}
        # node lables
        for i,j in enumerate(self.tree.feature):
            # if node is a decision it should be white
            if j != -2:
                labels[i] = str(feature_names[j]) + '\n'
                labels[i] +='>=' + str(np.round(self.tree.threshold[i],round_n))
                node_colors.append('white')
            # if node is terminal it should be colored
            # only works for binary decision trees
            else:
                # label = str(np.argmax(self.tree.value[i])) + '\n'
                color = self.tree.value[i][0][1]/np.sum(self.tree.value[i][0])
                purity[i] = str(int(np.sum(self.tree.value[i][0])))
                labels[i] = str(round(color,2))
                cix = round(color * pal_len,0)
                # the colors need to be scaled to 0 - 255 for bokeh
                # also there is an extra '1' at the end of each RGB code
                # that needs to be removed
                node_colors.append(tuple(palette[int(cix)][:3] * 255))

        # draw the nodes and labels
        for node in self.tree_map:
            y = self.tree_map[node]['y']
            x = self.tree_map[node]['x_abs']
            p.circle([x], [y], color = node_colors[node], radius = 0.4)
            p.text(x, y , [labels[node]], text_align = 'center',
                   text_font_size = '0.8em')

        # plot aesthetics
        p.grid.grid_line_color = None
        bplt.show(p)
