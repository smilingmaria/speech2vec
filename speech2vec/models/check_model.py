from seq2seq import *

batch_input_shape = (32, 60, 225)

cells = ['GRUCell','GRUCell']
hidden_dim = 128
depth = (2,2)

model = Seq2seq(batch_input_shape, cells, hidden_dim, depth, peek = True)

model.build_inputs()
model.build_encoder()
model.build_decoder()

cells = ['GRUCell'] * 3
model = AttentionSeq2seq(batch_input_shape, cells, hidden_dim, depth)

model.build_inputs()
model.build_encoder()
model.build_decoder()
