__author__ = 'Christopher Perrin'

from nltk import tree


class Converter(object):
    """
    A class to convert a dependency structure to a phrase structure

    :param dep_graph: A dependency graph to convert
    :type dep_graph: nltk.parse.dependencygraph.DependencyGraph
    :param proj_table: A projection table
    :type proj_table: dict
    :param arg_table: An argument table
    :type arg_table: dict
    :param mod_table: A modifier table
    :type mod_table: dict
    """
    def __init__(self, dep_graph, proj_table, arg_table, mod_table):

        self.dep_graph = dep_graph
        self.proj_table = proj_table
        self.arg_table = arg_table
        self.mod_table = mod_table

    def convert(self):
        """
        Starts the conversion to a phrase structure

        :return: The phrase structure tree as derived from the dependency tree
        :rtype: converter.PhraseTree
        """
        head = self._make_projection(self.dep_graph.root)
        deps = self.dep_graph.root['deps']
        self._convert_subtree(deps, head, self.dep_graph.root)
        return head

    def _convert_subtree(self, nodelist, suptree, supnode):
        print(nodelist, '\n', supnode)
        if len(nodelist) == 0:
            return

        left = list()

        while len(nodelist) > 0 and nodelist[0] < supnode['address']:
            left.insert(0, nodelist.pop(0))

        right = nodelist

        left_tree = suptree
        right_tree = suptree.copy(deep=True)
        for node in left:
            n = self.dep_graph.get_by_address(node)

            proj = self._make_projection(n)

            self._convert_subtree(n['deps'], proj, n)
            if n['rel'] == "ARG":
                left_tree = self._connect_arg_to_head(proj, left_tree, -1)
            elif n['rel'] == "MOD":
                left_tree = self._connect_mod_to_head(proj, left_tree, -1)

        for node in right:
            n = self.dep_graph.get_by_address(node)

            proj = self._make_projection(n)

            self._convert_subtree(n['deps'], proj, n)
            if n['rel'] == "ARG":
                right_tree = self._connect_arg_to_head(proj, right_tree, 1)
            elif n['rel'] == "MOD":
                right_tree = self._connect_mod_to_head(proj, right_tree, 1)

        n = suptree
        r = right_tree
        while len(n) > 0:
            new_n = n[len(n) - 1]
            while len(r) > 1:
                n.append(r.pop(1))

            n = new_n
            r = r[0]

    def _make_projection(self, node):
        """
        Projects a node according to the projection table.

        :param node: A node that should be projected according to the projection table
        :type node: converter.PhraseTree
        :return: A projected node
        :rtype: converter.PhraseTree
        """
        out = PhraseTree(node['word'], list())
        out = PhraseTree(node['tag'], [out])
        if not node['tag'] in self.proj_table.keys():
            return out

        for tag in self.proj_table[node['tag']]:
            out = PhraseTree(tag, [out])
        return out

    def _connect_arg_to_head(self, subtree, suptree, pos):
        head_node = suptree.find_fork()

        index = 1
        if pos == -1:
            index = 2

        while head_node is not None:
            if head_node.label() not in self.arg_table.keys() or self.arg_table[head_node.label()][index] == 0:
                head_node = head_node.parent()
                continue
            tree_node = subtree.find_fork()
            while tree_node is not None:
                if tree_node.label() in self.arg_table[head_node.label()][2]:
                    tree_node.clear_parent()
                    if pos == -1:
                        head_node.insert(0, tree_node)
                    else:
                        head_node.append(tree_node)

                    return suptree
                tree_node = tree_node.parent()

        raise ValueError("Can't find a connection to the head in the argument table")

    def _connect_mod_to_head(self, subtree, suptree, pos):
        head_node = suptree.find_fork()

        index = 1
        if pos == -1:
            index = 0

        while head_node is not None:
            if head_node.label() not in self.mod_table.keys():
                head_node = head_node.parent()
                continue
            tree_node = subtree.find_fork()
            while tree_node is not None:
                if tree_node.label() in self.mod_table[head_node.label()][index]:
                    tree_node.clear_parent()
                    if pos == -1:
                        head_node.insert(0, tree_node)
                    else:
                        head_node.append(tree_node)

                    return suptree
                tree_node = tree_node.parent()

        raise ValueError("Can't find a connection to the head in the modifier table")


class PhraseTree(tree.ParentedTree):
    def find_fork(self):
        if len(self) != 1:
            return self

        return self[0].find_fork()

    def clear_parent(self):
        self._parent = None
