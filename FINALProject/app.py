import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Dot, Activation, Concatenate
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QTextEdit, QPushButton, QMessageBox, QFrame, QGraphicsDropShadowEffect)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QTimer

# --- CONFIGURATION ---
MAX_SEQ_LEN = 50
LATENT_DIM = 256
EMBEDDING_DIM = 256
DEBOUNCE_DELAY = 1000  # Time in ms to wait (1 second)

class SpellCheckerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # UPDATED TITLE HERE
        self.setWindowTitle("Automatic spelling checker")
        self.resize(900, 700)
        
        # Setup the Debounce Timer (Real-time logic)
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.run_correction)
        
        self.init_ui()
        self.load_inference_model()

    def init_ui(self):
        # --- STYLING ---
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { font-family: 'Segoe UI', sans-serif; color: #E0E0E0; }
            QFrame#ContentFrame { background-color: #1E1E1E; border-radius: 15px; border: 1px solid #333; }
            QLabel#Title { font-size: 28px; font-weight: bold; color: #ffffff; }
            QLabel#Subtitle { font-size: 14px; color: #BBBBBB; margin-bottom: 10px; }
            QLabel#SectionLabel { font-size: 14px; font-weight: 600; color: #4FC3F7; margin-top: 10px; }
            QTextEdit { 
                background-color: #2C2C2C; border: 2px solid #3E3E3E; border-radius: 10px; 
                padding: 12px; font-size: 16px; color: #FFFFFF; 
            }
            QTextEdit:focus { border: 2px solid #4FC3F7; }
            QLabel#Status { color: #9E9E9E; font-size: 12px; padding: 5px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()
        
        # UPDATED APP NAME HERE
        title = QLabel("Spellchecker")
        title.setObjectName("Title")
        
        subtitle = QLabel("Automatic correction using Deep Learning")
        subtitle.setObjectName("Subtitle")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)

        # Content Frame
        content_frame = QFrame()
        content_frame.setObjectName("ContentFrame")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 5)
        content_frame.setGraphicsEffect(shadow)

        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(15)

        # Input
        lbl_input = QLabel("INPUT TEXT")
        lbl_input.setObjectName("SectionLabel")
        content_layout.addWidget(lbl_input)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Start typing here... (English or Hindi)")
        self.input_box.setFixedHeight(120)
        
        # Real-time connection
        self.input_box.textChanged.connect(self.on_user_typing)
        
        content_layout.addWidget(self.input_box)

        # Output
        lbl_output = QLabel("CORRECTED RESULT")
        lbl_output.setObjectName("SectionLabel")
        content_layout.addWidget(lbl_output)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setFixedHeight(120)
        self.output_box.setPlaceholderText("Waiting for input...")
        content_layout.addWidget(self.output_box)

        main_layout.addWidget(content_frame)

        # Status Bar
        self.status_label = QLabel("System Ready")
        self.status_label.setObjectName("Status")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

    def on_user_typing(self):
        self.status_label.setText("Typing...")
        self.debounce_timer.start(DEBOUNCE_DELAY)

    def load_inference_model(self):
        try:
            self.status_label.setText("Initializing AI...")
            QApplication.processEvents()

            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.idx2char = {v: k for k, v in self.tokenizer.word_index.items()}
            self.char2idx = self.tokenizer.word_index

            # Re-define Architecture
            enc_inputs = Input(shape=(MAX_SEQ_LEN,))
            enc_emb = Embedding(self.vocab_size, EMBEDDING_DIM)(enc_inputs)
            enc_lstm = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, return_state=True))
            enc_outputs, f_h, f_c, b_h, b_c = enc_lstm(enc_emb)
            state_h = Concatenate()([f_h, b_h])
            state_c = Concatenate()([f_c, b_c])
            enc_states = [state_h, state_c]

            dec_inputs = Input(shape=(MAX_SEQ_LEN,))
            dec_emb_layer = Embedding(self.vocab_size, EMBEDDING_DIM)
            dec_emb = dec_emb_layer(dec_inputs)
            dec_lstm = LSTM(LATENT_DIM * 2, return_sequences=True, return_state=True)
            dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=enc_states)

            attention = Dot(axes=[2, 2])([dec_outputs, enc_outputs])
            attention = Activation('softmax')(attention)
            context = Dot(axes=[2, 1])([attention, enc_outputs])
            decoder_combined_context = Concatenate()([context, dec_outputs])

            decoder_dense = Dense(self.vocab_size, activation='softmax')
            output = decoder_dense(decoder_combined_context)

            model_train = Model([enc_inputs, dec_inputs], output)
            model_train.load_weights('scmil_spell_corrector.h5')

            # Inference Models
            self.encoder_model = Model(enc_inputs, [enc_outputs] + enc_states)

            dec_state_input_h = Input(shape=(LATENT_DIM * 2,))
            dec_state_input_c = Input(shape=(LATENT_DIM * 2,))
            dec_states_inputs = [dec_state_input_h, dec_state_input_c]
            dec_input_single = Input(shape=(1,))
            dec_emb2 = dec_emb_layer(dec_input_single)
            enc_outputs_input = Input(shape=(MAX_SEQ_LEN, LATENT_DIM * 2))

            dec_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inputs)
            att2 = Dot(axes=[2, 2])([dec_outputs2, enc_outputs_input])
            att2 = Activation('softmax')(att2)
            context2 = Dot(axes=[2, 1])([att2, enc_outputs_input])
            dec_combined2 = Concatenate()([context2, dec_outputs2])
            output2 = decoder_dense(dec_combined2)

            self.decoder_model = Model(
                [dec_input_single, enc_outputs_input] + dec_states_inputs,
                [output2, state_h2, state_c2]
            )

            self.status_label.setText("Ready.")

        except Exception as e:
            self.status_label.setText("Error Loading Model")
            QMessageBox.critical(self, "Error", f"Model Error: {str(e)}")

    def decode_sequence(self, input_text):
        # Beam Search parameters
        BEAM_WIDTH = 3  # Looks at top 3 possibilities at once
        
        # 1. Encode
        seq = self.tokenizer.texts_to_sequences([input_text])
        seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
        enc_out, state_h, state_c = self.encoder_model.predict(seq, verbose=0)
        
        # 2. Initialize Beam
        # Each item in beam is a tuple: (current_seq, score, state_h, state_c, decoded_string)
        start_token = self.char2idx['\t']
        start_seq = np.zeros((1, 1))
        start_seq[0, 0] = start_token
        
        # Beam structure: [ (score, decoder_input, state_h, state_c, result_string) ]
        beam = [(0.0, start_seq, state_h, state_c, "")]
        
        # 3. Loop
        for _ in range(MAX_SEQ_LEN):
            candidates = []
            
            # Expand each candidate in the beam
            for score, target_seq, h, c, decoded_str in beam:
                if decoded_str.endswith('\n') or len(decoded_str) >= MAX_SEQ_LEN:
                    candidates.append((score, target_seq, h, c, decoded_str))
                    continue
                
                # Predict next char
                output_tokens, new_h, new_c = self.decoder_model.predict(
                    [target_seq, enc_out, h, c], verbose=0
                )
                
                # Get top k probabilities
                probs = output_tokens[0, -1, :]
                top_k_indices = np.argsort(probs)[-BEAM_WIDTH:]
                
                for idx in top_k_indices:
                    prob = probs[idx]
                    # Log probability for numerical stability (scores are negative, closer to 0 is better)
                    new_score = score + np.log(prob + 1e-10)
                    char = self.idx2char.get(idx, '')
                    
                    new_target_seq = np.zeros((1, 1))
                    new_target_seq[0, 0] = idx
                    
                    candidates.append((new_score, new_target_seq, new_h, new_c, decoded_str + char))
            
            # Sort candidates by score and keep top k
            ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
            beam = ordered[:BEAM_WIDTH]
            
            # Early stopping: if best candidate ended, we can stop
            if beam[0][4].endswith('\n'):
                break
                
        # Return the string of the best candidate
        best_candidate = beam[0][4]
        return best_candidate.strip()

    def run_correction(self):
        text = self.input_box.toPlainText().strip()
        if not text: return
        
        self.status_label.setText("Checking...")
        QApplication.processEvents()
        
        try:
            corrected = self.decode_sequence(text)
            self.output_box.setText(corrected)
            self.status_label.setText("Updated.")
        except Exception as e:
            self.status_label.setText("Error.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellCheckerApp()
    window.show()
    sys.exit(app.exec())
