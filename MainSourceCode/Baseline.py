"""
Play Tasks group
Implementation of the baseline 3-task MTL model/baseline 5-task MTL model/baseline 9-task MTL model

Interactive Tasks group
Implementation of the baseline 6-task MTL model
"""


class PLE(Estimator):
    def __init__(self,
                 feature_columns=,
                 embedding_size=,
                 task_names=,
                 loss_types=,
                 focal_info=,
                 multi_task_weights=,
                 gate_layers=,
                 tower_layers=,
                 fl_shared_expert_layers=,
                 fl_task_expert_layers=,
                 sl_shared_expert_layers=,
                 sl_task_expert_layers=,
                 use_dropout,
                 dropout_gate,
                 dropout_tower,
                 batch_norm,
                 batch_norm_decay,
                 uniform_param,
                 normal_param=,
                 optimizer,
                 activation,
                 multi_label_slotid,
                 flag_begin_key,
                 debug,
                 gate_version,
                 weight_info,
                 product_param,
                 se_param,
                 shared_bottom_layers,
                 is_user_simplify,
                 user_slots,
                 item_slots,
                 is_mtl_joint,
                 dump_sparse_opt_args,
                 is_train):
        assert (len(feature_columns)), \
            "feature_columns can not be empty"
        assert len(task_names) == len(loss_types), \
            "the number of tasks must be the same as the number of loss types"

        super(PLE, self).__init__()

        self.model_name = 'PLE'
        self.embedding_size = embedding_size
        self.task_names = task_names
        self.loss_types = loss_types
        self.gate_layers = gate_layers
        self.tower_layers = tower_layers
        self.shared_bottom_layers = shared_bottom_layers
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.dropout_gate = dropout_gate
        self.dropout_tower = dropout_tower
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.normal_param = normal_param
        self.uniform_param = uniform_param
        self.feature_columns = feature_columns  # slot id list
        self.multi_label_slotid = multi_label_slotid
        self.initializer = initializer
        self.activation = activation
        self.debug = debug
        self.gate_version = gate_version
        self.flag_begin_key = flag_begin_key
        self.weight_info = weight_info
        self.multi_task_weights = multi_task_weights
        self.is_mtl_joint = is_mtl_joint
        self.dump_sparse_opt_args = dump_sparse_opt_args
        self.is_train = is_train

        self.fl_shared_expert_layers = fl_shared_expert_layers
        self.fl_task_expert_layers = fl_task_expert_layers
        self.sl_shared_expert_layers = sl_shared_expert_layers
        self.sl_task_expert_layers = sl_task_expert_layers

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
        
        # 5.build expert modules
        self._build_first_level_input_component()
        self._build_first_level_expert_component()
        self._build_first_level_gate_component()

        self._build_second_level_expert_component()
        self._build_second_level_gate_component()

        # 6.build towers
        self._build_multi_towers_component()

        # 7.get final loss
        loss = tf.concat(self.task_losses.values(), axis=1)
        nonzero = tf.cast(tf.count_nonzero(loss), dtype=tf.float32)
        self.loss = tf.div(tf.reduce_sum(loss), nonzero)

    def _build_first_level_input_component(self):
        self.first_level_input_layers = {}
        middle_layer_size = int(self.embedding_layer.shape[-1])
        # shared
        if self.se_param and self.se_param["use_first_level_se"]:
            self.first_level_input_layers["shared_component"] = self._build_se_layer(self.embedding_layer,
                                                                                     middle_layer_size,
                                                                                     name="shared_component")
        else:
            self.first_level_input_layers["shared_component"] = self.embedding_layer

        # task specific
        for index, task_name in enumerate(self.task_names):
            if self.se_param and self.se_param["use_first_level_se"]:
                self.first_level_input_layers[task_name] = self._build_se_layer(self.embedding_layer,
                                                                                middle_layer_size,
                                                                                name=task_name)
            else:
                self.first_level_input_layers[task_name] = self.embedding_layer

    def _build_first_level_expert_component(self):
        self.first_expert_layers = {}
        # shared
        input_layer = self.first_level_input_layers["shared_component"]
        first_shared_expert_layers = self._build_expert_component(input_layer, self.fl_shared_expert_layers,
                                                                  name="first_shared")
        self.first_expert_layers["shared_component"] = first_shared_expert_layers

        # task specific
        for index, task_name in enumerate(self.task_names):
            input_layer = self.first_level_input_layers[task_name]
            first_task_expert_layers = \
                self._build_expert_component(input_layer, self.fl_task_expert_layers,
                                             name="first_task_{}".format(task_name))
            self.first_expert_layers[task_name] = first_task_expert_layers

    def _build_first_level_gate_component(self):
        # shared
        gate_input_component_list = []
        for task_name in self.task_names:
            gate_input_component_list.extend(self.first_expert_layers[task_name])
        gate_input_component_list.extend(self.first_expert_layers["shared_component"])
        shared_expert_number = len(gate_input_component_list)

        gate_weighted_layer = self._build_gate_weighted_component(self.first_level_input_layers["shared_component"],
                                                                  shared_expert_number,
                                                                  name="first_shared_component")  # [batchsize, expert_number]

        self.first_shared_gate_output_layer = self._build_gate_output_component(shared_expert_number,
                                                                                gate_input_component_list,
                                                                                gate_weighted_layer,
                                                                                name="first_shared_component")

        # task specific
        self.first_task_gate_output_layer = {}
        for task_name in self.task_names:
            gate_input_component_list = []
            gate_input_component_list.extend(self.first_expert_layers[task_name])
            gate_input_component_list.extend(self.first_expert_layers["shared_component"])
            task_specific_expert_number = len(gate_input_component_list)

            gate_weighted_layer = self._build_gate_weighted_component(self.first_level_input_layers[task_name],
                                                                      task_specific_expert_number,
                                                                      name="first_{}".format(
                                                                          task_name))  # [batchsize, expert_number]

            self.first_task_gate_output_layer[task_name] = \
                self._build_gate_output_component(task_specific_expert_number, gate_input_component_list,
                                                  gate_weighted_layer,
                                                  name="first_{}".format(task_name))

    def _build_second_level_expert_component(self):
        self.second_expert_layers = {}
        # shared
        input_layer = self.first_shared_gate_output_layer
        second_shared_expert_layers = self._build_expert_component(input_layer, self.sl_shared_expert_layers,
                                                                   name="second_shared")
        self.second_expert_layers["shared_component"] = second_shared_expert_layers

        # task specific
        for index, task_name in enumerate(self.task_names):
            input_layer = self.first_task_gate_output_layer[task_name]
            second_task_expert_layers = \
                self._build_expert_component(input_layer, self.sl_task_expert_layers,
                                             name="second_task_{}".format(task_name))
            self.second_expert_layers[task_name] = second_task_expert_layers

    def _build_second_level_gate_component(self):
        # task specific
        self.second_task_gate_output_layer = {}
        for task_name in self.task_names:
            gate_input_component_list = []
            gate_input_component_list.extend(self.second_expert_layers[task_name])
            gate_input_component_list.extend(self.second_expert_layers["shared_component"])
            task_specific_expert_number = len(gate_input_component_list)

            gate_weighted_layer = self._build_gate_weighted_component(self.first_task_gate_output_layer[task_name],
                                                                      task_specific_expert_number,
                                                                      name="second_{}".format(task_name))

            self.second_task_gate_output_layer[task_name] = \
                self._build_gate_output_component(task_specific_expert_number, gate_input_component_list,
                                                  gate_weighted_layer,
                                                  name="second_{}".format(task_name))

    def _build_gate_output_component(self, expert_number, gate_input_component_list, gate_weighted_layer, name=""):
        for index in range(expert_number):
            gate_weight = tf.slice(gate_weighted_layer, [0, index], [-1, 1],
                                   name="{0}_gate_slice_{1}".format(name, index))
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
        expert_output_layers = []
        expert_num = len(expert_layers)
        self.simplify_mlp[name] = dict()
        for expert_index in range(expert_num):
            expert_def = expert_layers[expert_index]
            if expert_def["name"] == "mlp":
                if self.user_simplify_func == "expert" and name[:5] == "first":
                    dnn = self._build_user_simplify_layer(expert_def["layers"], expert_index, name)
                    expert_output_layers.append(dnn)
                else:
                    dnn = input_layer
                    for layer_index in range(len(expert_def["layers"])):
                        output_size = expert_def["layers"][layer_index]
                        dnn = self._build_mlp_layer(input_layer=dnn, output_size=output_size,
                                                    activation=self.activation, direct_output=False,
                                                    name="expert_{0}_{1}_layer_{2}".format(name, expert_index, layer_index))
                    expert_output_layers.append(dnn)
        return expert_output_layers

    def _build_gate_weighted_component(self, gate_input_layer, gate_output_size, name=""):
        print("gate_output_size:{}".format(gate_output_size))
        if len(self.gate_layers) > 0:
            for layer_index in range(len(self.gate_layers)):
                output_size = self.gate_layers[layer_index]
                gate_input_layer = self._build_mlp_layer(input_layer=gate_input_layer,
                                                         output_size=output_size,
                                                         activation=self.activation, direct_output=False,
                                                         name="gate_{0}_layer_{1}".format(name, layer_index))

        gate_output_layer = self._build_mlp_layer(input_layer=gate_input_layer,
                                                  output_size=gate_output_size,
                                                  activation=tf.nn.softmax, direct_output=True,
                                                  name="gate_{0}_output".format(name))
        return gate_output_layer

    def _build_tower_component(self, task_name=""):
        tower_input_layer = self.second_task_gate_output_layer[task_name]

        if len(self.tower_layers) > 0:
            for layer_index in range(len(self.tower_layers)):
                output_size = self.tower_layers[layer_index]
                tower_input_layer = self._build_mlp_layer(input_layer=tower_input_layer,
                                                          output_size=output_size,
                                                          activation=self.activation, direct_output=False,
                                                          name="tower_{0}_layer_{1}".format(task_name, layer_index))
        logits = self._build_mlp_layer(input_layer=tower_input_layer, output_size=1,
                                       activation=None, direct_output=True,
                                       name="tower_{0}".format(task_name))
        return logits

    def _build_multi_towers_component(self):
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

    def _build_mlp_layer(self, input_layer, output_size, direct_output=False, activation=None, name="name"):
        output_layer = tf.layers.dense(inputs=input_layer, units=output_size,
                                       use_bias=True, activation=activation,
                                       kernel_initializer=self.initializer, bias_initializer=self.initializer,
                                       name=name)
        if not direct_output:
            if self.batch_norm:
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
