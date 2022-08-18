"""
Play Tasks group
Implementation of the MFH 9-task MTL model

Interactive Tasks group
Implementation of the MFH 12-task MTL model
"""

class MFH(Estimator):
    def __init__(self,
                 feature_columns,
                 embedding_size,
                 task_names,
                 loss_types,
                 multi_task_weights,
                 first_level_param,
                 second_level_param,
                 batch_norm,
                 batch_norm_decay,
                 optimizer,
                 activation,
                 initializer,
                 multi_label_slotid,
                 flag_begin_key,
                 debug,
                 weight_info,
                 is_mtl_joint,
                 dump_sparse_opt_args,
                 is_train,
                 use_sparse_placeholder):
        assert (len(feature_columns)), \
            "feature_columns can not be empty"
        assert len(task_names) == len(loss_types), \
            "the number of tasks must be the same as the number of loss types"

        super(MFH, self).__init__()

        self.model_name = 'MFH'
        self.embedding_size = embedding_size
        self.task_names = task_names
        self.loss_types = loss_types
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.feature_columns = feature_columns
        self.multi_label_slotid = multi_label_slotid
        self.initializer = initializer
        self.activation = activation
        self.debug = debug
        self.flag_begin_key = flag_begin_key
        self.weight_info = weight_info
        self.multi_task_weights = multi_task_weights
        self.is_mtl_joint = is_mtl_joint
        self.dump_sparse_opt_args = dump_sparse_opt_args
        self.is_train = is_train

        self.first_level_name = first_level_param.model_name
        self.second_level_name = second_level_param.model_name
        self.first_level_tower_layers = first_level_param.tower_layers
        self.second_level_tower_layers = second_level_param.tower_layers

        self.first_level_hard_sharing_layers = first_level_param.hard_sharing_layers.hidden_units
        self.second_level_hard_sharing_layers = second_level_param.hard_sharing_layers.hidden_units

        self.gate_version = dict()
        self.gate_layers = dict()
        self.gate_version["first_level"] = first_level_param.gate_version
        self.gate_layers["first_level"] = first_level_param.gate_layers.hidden_units
        self.gate_version["second_level"] = second_level_param.gate_version
        self.gate_layers["second_level"] = second_level_param.gate_layers.hidden_units

        self.first_level_param = first_level_param
        self.second_level_param = second_level_param

        self.first_level_mtl_output = {}
        self.second_level_mtl_output = {}

        self.fl_shared_expert_layers = {}
        self.fl_task_expert_layers = {}
        self.sl_shared_expert_layers = {}
        self.sl_task_expert_layers = {}

        self.ple_fl_expert_layers = {}
        self.ple_sl_expert_layers = {}
        self.ple_fl_shared_gate_output_layers = {}
        self.ple_fl_task_gate_output_layers = {}
        self.ple_sl_task_gate_output_layers = {}

        self._initialize_task_type()
        self._initialize_expert_param()
        self.use_sparse_placeholder = use_sparse_placeholder

        print(self.__dict__.items())

        self._build_graph()

    def _build_graph(self):
        # 1.initialize label placeholder
        self.label = tf.placeholder(tf.float32, name="label")

        # 2.initialize control placeholder
        self._initialize_control_placeholder()

        # 3.initialize task flag placeholder
        self.initialize_flag_placeholder()

        # 4.initialize embedding lookup table
        self._initialize_embeddings()

        # 5.build multi-faceted hierarchical struct
        self._build_first_level_component()
        self._build_second_level_component()

        # 6.choose final prediction and loss according to task flag
        self._get_final_loss()

    def _initialize_task_type(self):
        self.task_name_to_user_action_map = {}
        self.task_name_to_user_group_map = {}
        self.user_actions = []
        self.user_groups = []

        for task_name in self.task_names:
            task_name_split = task_name.split("_")
            assert (len(task_name_split) >= 2), "error task_name:{}".format(task_name)
            user_action = '_'.join(task_name_split[0:-1])
            user_group = task_name_split[-1]
            self.task_name_to_user_action_map[task_name] = user_action
            self.task_name_to_user_group_map[task_name] = user_group

            if user_action not in self.user_actions:
                self.user_actions.append(user_action)
            if user_group not in self.user_groups:
                self.user_groups.append(user_group)

        print("task_name_to_user_action_map:{}".format(self.task_name_to_user_action_map))
        print("task_name_to_user_group_map:{}".format(self.task_name_to_user_group_map))
        print("user_actions:{}".format(self.user_actions))
        print("user_groups:{}".format(self.user_groups))

    def _initialize_expert_param(self):
        for param in [self.first_level_param, self.second_level_param]:
            all_expert_layers = [[], [], [], []]
            for index, expert_layers_config in \
                    enumerate([param.fl_shared_expert_layers, param.fl_task_expert_layers,
                               param.sl_shared_expert_layers, param.sl_task_expert_layers]):
                for expert_layer_param in expert_layers_config:
                    expert_layer_def = {"name": expert_layer_param.name}
                    layers = expert_layer_param.hidden_units
                    if len(layers) > 0:
                        expert_layer_def["layers"] = layers
                    if expert_layer_param.HasField("dropout_parameter"):
                        dropout = expert_layer_param.dropout_parameter.dropout_rate
                        if len(dropout) > 0:
                            expert_layer_def["dropout"] = dropout
                    all_expert_layers[index].append(expert_layer_def)

            self.fl_shared_expert_layers[param.level_name] = all_expert_layers[0]
            self.fl_task_expert_layers[param.level_name] = all_expert_layers[1]
            self.sl_shared_expert_layers[param.level_name] = all_expert_layers[2]
            self.sl_task_expert_layers[param.level_name] = all_expert_layers[3]

    def _build_first_level_component(self):

        if self.first_level_name == "PLE":
            print("first_level use PLE")

            component_name = "first_level_user_action"
            self.first_level_mtl_output[component_name] = self._build_ple_component("first_level",
                                                                                    self.embedding_layer,
                                                                                    self.user_actions,
                                                                                    component_name)
            component_name = "first_level_user_group"
            self.first_level_mtl_output[component_name] = self._build_ple_component("first_level",
                                                                                    self.embedding_layer,
                                                                                    self.user_groups,
                                                                                    component_name)

        elif self.first_level_name == "CGC":
            print("first_level use CGC")
            component_name = "first_level_user_action"
            self.first_level_mtl_output[component_name] = self._build_cgc_component("first_level",
                                                                                    self.embedding_layer,
                                                                                    self.user_actions,
                                                                                    component_name)
            component_name = "first_level_user_group"
            self.first_level_mtl_output[component_name] = self._build_cgc_component("first_level",
                                                                                    self.embedding_layer,
                                                                                    self.user_groups,
                                                                                    component_name)

        print("after _build_first_level_component self.first_level_mtl_output:{}".format(self.first_level_mtl_output))

    def _build_second_level_component(self):
        self.second_level_tower_input_layers = {}
        if self.second_level_name == "PLE":
            print("second_level use PLE")
            for user_action in self.user_actions:
                component_name = "second_level_user_action_" + user_action
                self.second_level_mtl_output[component_name] = \
                    self._build_ple_component("second_level",
                                              self.first_level_mtl_output["first_level_user_action"][user_action],
                                              self.user_groups,
                                              component_name)

            for user_group in self.user_groups:
                component_name = "second_level_user_group_" + user_group
                self.second_level_mtl_output[component_name] = \
                    self._build_ple_component("second_level",
                                              self.first_level_mtl_output["first_level_user_group"][user_group],
                                              self.user_actions,
                                              component_name)

        elif self.second_level_name == "CGC":
            print("second_level use CGC")
            for user_action in self.user_actions:
                component_name = "second_level_user_action_" + user_action
                self.second_level_mtl_output[component_name] = \
                    self._build_cgc_component("second_level",
                                              self.first_level_mtl_output["first_level_user_action"][user_action],
                                              self.user_groups,
                                              component_name)

            for user_group in self.user_groups:
                component_name = "second_level_user_group_" + user_group
                self.second_level_mtl_output[component_name] = \
                    self._build_cgc_component("second_level",
                                              self.first_level_mtl_output["first_level_user_group"][user_group],
                                              self.user_actions,
                                              component_name)
        print(
            "after _build_second_level_component self.second_level_mtl_output:{}".format(self.second_level_mtl_output))

        self._build_second_level_tower_component()

    def _build_ple_component(self, level_name, input_layer, tasks, component_name):
        print("_build_ple_component level_name:{}, input_layer:{}, tasks:{}, component_name:{}".format(level_name,
                                                                                                       input_layer,
                                                                                                       tasks,
                                                                                                       component_name))
        self._build_ple_fl_expert_component(level_name, input_layer, tasks, component_name)
        self._build_ple_fl_gate_component(level_name, input_layer, tasks, component_name)

        self._build_ple_sl_expert_component(level_name, tasks, component_name)
        self._build_ple_sl_gate_component(level_name, tasks, component_name)

        return self.ple_sl_task_gate_output_layers[component_name]

    def _build_cgc_component(self, level_name, input_layer, tasks, component_name):
        print("_build_cgc_component input_layer:{}, tasks:{}, component_name:{}".format(input_layer, tasks,
                                                                                        component_name))
        self._build_ple_fl_expert_component(level_name, input_layer, tasks, component_name)
        self._build_ple_fl_gate_component(level_name, input_layer, tasks, component_name)

        return self.ple_fl_task_gate_output_layers[component_name]

    def _build_ple_fl_expert_component(self, level_name, input_layer, tasks, component_name):
        print("_build_ple_fl_expert_component input_layer:{}, tasks:{}, component_name:{}".format(input_layer, tasks,
                                                                                                  component_name))
        if component_name not in self.ple_fl_expert_layers:
            self.ple_fl_expert_layers[component_name] = {}

        # shared
        fl_shared_expert_layers = self._build_expert_component(input_layer,
                                                               self.fl_shared_expert_layers[level_name],
                                                               name="{}_first_shared".format(component_name))
        self.ple_fl_expert_layers[component_name]["shared_component"] = fl_shared_expert_layers

        # task specific
        for task_name in tasks:
            fl_task_expert_layers = \
                self._build_expert_component(input_layer,
                                             self.fl_task_expert_layers[level_name],
                                             name="{}_first_task_{}".format(component_name, task_name))
            self.ple_fl_expert_layers[component_name][task_name] = fl_task_expert_layers

    def _build_ple_fl_gate_component(self, level_name, input_layer, tasks, component_name):
        print("_build_ple_fl_gate_component input_layer:{}, tasks:{}, component_name:{}".format(input_layer, tasks,
                                                                                                component_name))

        # shared
        gate_input_component_list = []
        for task_name in tasks:
            gate_input_component_list.extend(self.ple_fl_expert_layers[component_name][task_name])

        gate_input_component_list.extend(self.ple_fl_expert_layers[component_name]["shared_component"])
        shared_expert_number = len(gate_input_component_list)

        gate_weighted_layer = self._build_gate_weighted_component(input_layer,
                                                                  shared_expert_number,
                                                                  level_name,
                                                                  name="{}_first_shared_component".format(
                                                                      component_name))

        self.ple_fl_shared_gate_output_layers[component_name] = \
            self._build_gate_output_component(shared_expert_number,
                                              gate_input_component_list,
                                              gate_weighted_layer,
                                              level_name,
                                              name="{}_first_shared_component".format(
                                                  component_name))

        # task specific
        if level_name not in self.ple_fl_task_gate_output_layers:
            self.ple_fl_task_gate_output_layers[component_name] = {}

        for task_name in tasks:
            gate_input_component_list = []
            gate_input_component_list.extend(self.ple_fl_expert_layers[component_name][task_name])
            gate_input_component_list.extend(self.ple_fl_expert_layers[component_name]["shared_component"])

            task_specific_expert_number = len(gate_input_component_list)

            gate_weighted_layer = self._build_gate_weighted_component(input_layer,
                                                                      task_specific_expert_number,
                                                                      level_name,
                                                                      name="{}_first_{}".format(
                                                                          component_name, task_name))

            self.ple_fl_task_gate_output_layers[component_name][task_name] = \
                self._build_gate_output_component(task_specific_expert_number,
                                                  gate_input_component_list,
                                                  gate_weighted_layer,
                                                  level_name,
                                                  name="{}_first_{}".format(component_name, task_name))

    def _build_ple_sl_expert_component(self, level_name, tasks, component_name):
        print("_build_ple_sl_expert_component level_name:{}, tasks:{}, component_name:{}".format(level_name, tasks,
                                                                                                 component_name))

        # shared
        if level_name not in self.ple_sl_expert_layers:
            self.ple_sl_expert_layers[component_name] = {}

        input_layer = self.ple_fl_shared_gate_output_layers[component_name]
        second_shared_expert_layers = self._build_expert_component(input_layer,
                                                                   self.sl_shared_expert_layers[level_name],
                                                                   name="{}_second_shared".format(component_name))
        self.ple_sl_expert_layers[component_name]["shared_component"] = second_shared_expert_layers

        # task specific
        for task_name in tasks:
            input_layer = self.ple_fl_task_gate_output_layers[component_name][task_name]
            second_task_expert_layers = \
                self._build_expert_component(input_layer, self.sl_task_expert_layers[level_name],
                                             name="{}_second_task_{}".format(component_name, task_name))
            self.ple_sl_expert_layers[component_name][task_name] = second_task_expert_layers

    def _build_ple_sl_gate_component(self, level_name, tasks, component_name):
        print("_build_ple_sl_gate_component level_name:{}, tasks:{}, component_name:{}".format(level_name, tasks,
                                                                                               component_name))

        # task specific
        if level_name not in self.ple_sl_task_gate_output_layers:
            self.ple_sl_task_gate_output_layers[component_name] = {}

        for task_name in tasks:
            gate_input_component_list = []
            gate_input_component_list.extend(self.ple_sl_expert_layers[component_name][task_name])
            gate_input_component_list.extend(self.ple_sl_expert_layers[component_name]["shared_component"])
            task_specific_expert_number = len(gate_input_component_list)

            gate_weighted_layer = \
                self._build_gate_weighted_component(self.ple_fl_task_gate_output_layers[component_name][task_name],
                                                    task_specific_expert_number,
                                                    level_name,
                                                    name="{}_second_{}".format(component_name, task_name))

            self.ple_sl_task_gate_output_layers[component_name][task_name] = \
                self._build_gate_output_component(task_specific_expert_number,
                                                  gate_input_component_list,
                                                  gate_weighted_layer,
                                                  level_name,
                                                  name="{}_second_{}".format(component_name, task_name))

    def _build_gate_output_component(self, expert_number, gate_input_component_list, gate_weighted_layer, level_name,
                                     name=""):
        print("_build_gate_output_component, expert_number:{} gate_input_component_list:{} gate_weighted_layer:{} "
              "level_name:{} name:{}".format(expert_number, gate_input_component_list, gate_weighted_layer, level_name,
                                             name))
        print(self.embedding_size)
        for index in range(expert_number):
            gate_weight = tf.slice(gate_weighted_layer, [0, index], [-1, 1],
                                   name="{0}_gate_slice_{1}".format(name, index))
            if self.gate_version[level_name] == 1:
                gate_weight = tf.multiply(gate_weight, 1 / expert_number)

            print("gate_weight shape:{0}, index:{1}, name:{2}".format(gate_weight.shape, index, name))
            if index == 0:
                gate_output_layer = tf.multiply(gate_input_component_list[index], gate_weight,
                                                name="{0}_expert_gate_mul_{1}".format(name, index))
            else:
                gate_output_layer = tf.add(gate_output_layer,
                                           tf.multiply(gate_input_component_list[index], gate_weight,
                                                       name="{0}_expert_gate_mul_{1}".format(name, index)),
                                           name="{0}_expert_gate_add_{1}".format(name, index))

        return gate_output_layer

    def _build_expert_component(self, input_layer, expert_layers, name=""):
        print("_build_expert_component input_layer:{}, expert_layers:{}, name:{}".format(input_layer, expert_layers,
                                                                                         name))
        expert_output_layers = []
        expert_num = len(expert_layers)
        for expert_index in range(expert_num):
            expert_def = expert_layers[expert_index]
            if expert_def["name"] == "mlp":
                dnn = input_layer
                for layer_index in range(len(expert_def["layers"])):
                    output_size = expert_def["layers"][layer_index]
                    dnn = self._build_mlp_layer(input_layer=dnn, output_size=output_size,
                                                activation=self.activation, direct_output=False,
                                                name="expert_{0}_{1}_layer_{2}".format(name, expert_index, layer_index))
                expert_output_layers.append(dnn)
        return expert_output_layers

    def _build_gate_weighted_component(self, gate_input_layer, gate_output_size, level_name, name=""):
        print("_build_gate_weighted_component, gate_input_layer:{} gate_output_size:{} level_name:{} name:{}".format(
            gate_input_layer, gate_output_size, level_name, name))

        gate_layers = self.gate_layers[level_name]
        if len(gate_layers) > 0:
            # process mlps before gate
            for layer_index in range(len(gate_layers)):
                output_size = gate_layers[layer_index]
                gate_input_layer = self._build_mlp_layer(input_layer=gate_input_layer,
                                                         output_size=output_size,
                                                         activation=self.activation, direct_output=False,
                                                         name="gate_{0}_layer_{1}".format(name, layer_index))
        if self.gate_version[level_name] == 1:
            gate_output_layer = self._build_mlp_layer(input_layer=gate_input_layer,
                                                      output_size=gate_output_size,
                                                      activation=tf.nn.sigmoid, direct_output=True,
                                                      name="gate_{0}_output".format(name))
        else:
            gate_output_layer = self._build_mlp_layer(input_layer=gate_input_layer,
                                                      output_size=gate_output_size,
                                                      activation=tf.nn.softmax, direct_output=True,
                                                      name="gate_{0}_output".format(name))
        return gate_output_layer

    def _build_tower_component(self, task_name=""):
        user_group = self.task_name_to_user_group_map[task_name]
        user_action = self.task_name_to_user_action_map[task_name]
        print("_build_tower_component, task_name:{}, user_group:{}, user_action:{}".format(task_name, user_group,
                                                                                           user_action))

        component_name = "second_level_user_action_" + user_action
        tower_input_layer_user_group = self.second_level_mtl_output[component_name][user_group]

        component_name = "second_level_user_group_" + user_group
        tower_input_layer_user_action = self.second_level_mtl_output[component_name][user_action]
        print("tower_input_layer_user_group:{}, tower_input_layer_user_action:{}".format(tower_input_layer_user_group,
                                                                                         tower_input_layer_user_action))

        tower_input_layer = tf.add(tower_input_layer_user_group,
                                   tower_input_layer_user_action,
                                   name="{}_tower_input_emb_add".format(task_name))

        tower_layers_hidden_units = self.second_level_tower_layers[self.user_group_index_map[user_group]].hidden_units
        if len(tower_layers_hidden_units) > 0:
            for layer_index in range(len(tower_layers_hidden_units)):
                output_size = tower_layers_hidden_units[layer_index]
                tower_input_layer = self._build_mlp_layer(input_layer=tower_input_layer,
                                                          output_size=output_size,
                                                          activation=self.activation, direct_output=False,
                                                          name="final_tower_{0}_layer_{1}".format(task_name,
                                                                                                  layer_index))
        logits = self._build_mlp_layer(input_layer=tower_input_layer, output_size=1,
                                       activation=None, direct_output=True,
                                       name="final_tower_{0}".format(task_name))
        return logits

    def _build_second_level_tower_component(self):
        self.task_preds = dict()
        self.task_losses = dict()

        preds = []
        labels = []
        for index, task_name in enumerate(self.task_names):
            loss_type = self.loss_types[index]
            logits = self._build_tower_component(task_name=task_name)
            cur_label = self.joint_labels[task_name]
            if loss_type == "logloss":
                pred = tf.nn.sigmoid(logits, name=task_name)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=cur_label,
                                                               name="logloss_{0}".format(task_name))
            elif loss_type == "mse":
                pred = tf.concat([logits], axis=0, name=task_name)
                loss = tf.square(cur_label - logits, name="mse_loss_{0}".format(task_name))
            else:
                print("unknown loss type")
            preds.append(pred)
            labels.append(cur_label)

            # task weight loss
            loss = tf.Print(loss, [loss], message="{0} : {1} task raw loss: ".format(task_name, loss_type), first_n=1,
                            summarize=100)
            task_weight = self.multi_task_weights[index]
            if task_weight > 0.0:
                print("{0} use task weight:{1}".format(task_name, task_weight))
                loss = tf.multiply(loss, task_weight)
            loss = tf.Print(loss, [loss], message="{0} : {1} task final loss: ".format(task_name, loss_type), first_n=1,
                            summarize=100)

            loss = tf.multiply(loss, self.joint_masks[task_name])

            self.task_preds[task_name] = pred
            self.task_losses[task_name] = loss
        self.preds = tf.concat(preds, axis=1, name="preds")
        self.labels = tf.concat(labels, axis=1, name="labels")

    def _get_final_loss(self):
        loss = tf.concat(self.task_losses.values(), axis=1)
        nonzero = tf.cast(tf.count_nonzero(loss), dtype=tf.float32)
        self.loss = tf.div(tf.reduce_sum(loss), nonzero)

    def _build_mlp_layer(self, input_layer, output_size, direct_output=False, activation=None, name="name"):
        output_layer = tf.layers.dense(inputs=input_layer, units=output_size,
                                       use_bias=True, activation=activation,
                                       kernel_initializer=self.initializer,
                                       bias_initializer=self.initializer,
                                       name=name)
        if not direct_output and self.batch_norm:
            output_layer = tf.layers.batch_normalization(output_layer,
                                                         training=self.control_placeholder["is_training"],
                                                         momentum=self.batch_norm_decay,
                                                         name="{0}_batch_normalization".format(name))
        return output_layer


    def _initialize_embeddings(self):
        """
        internal code
        init embedding lookup table and embedding_layer
        :return: None
        """

    def _initialize_control_placeholder(self):
        self.control_placeholder = dict()

        # bn is_training control holder
        is_training = tf.placeholder(tf.bool, name='is_training')
        self.control_placeholder["is_training"] = is_training
