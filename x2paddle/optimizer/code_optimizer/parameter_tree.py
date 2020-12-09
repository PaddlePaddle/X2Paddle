# -*- coding:UTF-8 -*-
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PamareterNode(object):
    def __init__(self, old_name=None, new_name=None):
        self.old_name = old_name
        self.new_name = new_name
        self.childs = list()

    def add_child(self, child):
        self.childs.append(child)
        
    def has_child(self):
        if len(self.childs) == 0:
            return False
        else:
            return True
       
        
class PamareterTree(object):
    def __init__(self):
        self.nodes = list()
        self.old2new = dict()
        
    def add_node(self, node):
        self.nodes.append(node)
    
    def traverse(self):
        tmp = list()
        def recurs(node, prefix_name):
            for child in node.childs:
                child_prefix_name = prefix_name + "." + child.new_name
                if child.has_child():
                    recurs(child, child_prefix_name)
                else:
                    self.old2new[child.old_name] = child_prefix_name[1:]
        recurs(self.nodes[-1], "")

        
    def get_node(self, old_name):
        for node in self.nodes:
            if node.old_name == old_name:
                return node