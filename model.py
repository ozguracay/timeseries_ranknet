from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Subtract

TIME_STAMP = _
NUM_FEATURE = _
OUT_SPACE_DIM = _

left_input = Input(shape=(TIME_STAMP, NUM_FEATURE), name='left_input')

right_input = Input(shape=(TIME_STAMP, NUM_FEATURE), name='right_input')

# shared base network
lstm_encoder = LSTM(OUT_SPACE_DIM, return_state=True)
lstm_decoder = LSTM(OUT_SPACE_DIM, return_sequences=True, return_state=True)
lstm_flatten = Flatten()

# left network
_, state_h_l, state_c_l = lstm_encoder(left_input)
encoder_states_l = [state_h_l, state_c_l]
decoder_outputs_l, _, _ = lstm_decoder(left_input, initial_state=encoder_states_l)
flatted_decoder_l = lstm_flatten(decoder_outputs_l)

# right network
_, state_h_r, state_c_r = lstm_encoder(right_input)
encoder_states_r = [state_h_r, state_c_r]
decoder_outputs_r, _, _ = lstm_decoder(right_input, initial_state=encoder_states_r)
flatted_decoder_r = lstm_flatten(decoder_outputs_r)

# left - right
diff = Subtract()([flatted_decoder_l, flatted_decoder_r])

outputs = Dense(1, activation="sigmoid")(diff)

model = Model(inputs=[left_input, right_input], outputs=outputs)
