import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class TratamentoDeImagens:
    def __init__(self, janela_principal):
        self.janela_principal = janela_principal
        self.janela_principal.title("Tratamento de Imagens")
        
        self.botao_carregar = tk.Button(janela_principal, text="Carregar Imagem", command=self.carregar_imagem)
        self.botao_carregar.pack()
        
        self.botao_treinar = tk.Button(janela_principal, text="Treinar Modelo", command=self.treinar_modelo)
        self.botao_treinar.pack()
        
        self.canvas = tk.Canvas(janela_principal, width=500, height=500)
        self.canvas.pack()
        
        self.barra_progresso = ttk.Progressbar(janela_principal, orient="horizontal", length=400, mode="determinate")
        self.barra_progresso.pack(pady=10)
        
        self.label_status = tk.Label(janela_principal, text="Status: Aguardando imagem...", fg="blue")
        self.label_status.pack(pady=5)
        
        self.imagem_original = None
        self.imagem_processada = None
        self.modelo = self.criar_modelo()
        self.modelo_carregado = False
        self.carregar_modelo()  # Tenta carregar o modelo ao iniciar

    def criar_modelo(self):
        modelo = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return modelo

    def carregar_modelo(self):
        try:
            self.modelo = tf.keras.models.load_model('whale_tail_model.h5')
            self.modelo_carregado = True
            self.label_status.config(text="Status: Modelo carregado com sucesso!", fg="green")
        except Exception as e:
            self.label_status.config(text=f"Status: Erro ao carregar modelo - {str(e)}", fg="red")
        self.janela_principal.update_idletasks()

    def carregar_imagem(self):
        arquivo = filedialog.askopenfilename(filetypes=[("Imagem", ".jpg;.png")])
        if arquivo:
            self.label_status.config(text="Status: Carregando imagem...", fg="blue")
            self.janela_principal.update_idletasks()
            
            self.imagem_original = cv2.imread(arquivo)
            self.imagem_original = cv2.cvtColor(self.imagem_original, cv2.COLOR_BGR2RGB)
            self.exibir_imagem(self.imagem_original, "Imagem Original")
            self.tratar_imagem()

            if self.modelo_carregado:
                self.prever_baleia()

    def tratar_imagem(self):
        if self.imagem_original is not None:
            self.barra_progresso["value"] = 0
            self.label_status.config(text="Status: Redimensionando imagem...", fg="blue")
            self.janela_principal.update_idletasks()
            
            imagem_redimensionada = cv2.resize(self.imagem_original, (224, 224))
            self.exibir_imagem(imagem_redimensionada, "Imagem Redimensionada")
            
            self.barra_progresso["value"] = 66
            self.label_status.config(text="Status: Normalizando imagem...", fg="blue")
            self.janela_principal.update_idletasks()
            imagem_normalizada = imagem_redimensionada / 255.0
            self.exibir_imagem((imagem_normalizada * 255).astype(np.uint8), "Imagem Normalizada")
            
            self.barra_progresso["value"] = 100
            self.label_status.config(text="Status: Imagem processada com sucesso!", fg="green")
            self.janela_principal.update_idletasks()

            self.imagem_processada = np.expand_dims(imagem_normalizada, axis=0)

    def prever_baleia(self):
        if self.modelo_carregado and self.imagem_processada is not None:
            self.label_status.config(text="Status: Fazendo previsão...", fg="blue")
            self.janela_principal.update_idletasks()
            
            previsao = self.modelo.predict(self.imagem_processada)
            resultado = "Mesma Baleia" if previsao > 0.5 else "Baleias Diferentes"
            self.exibir_resultado(resultado)
            self.label_status.config(text="Status: Previsão concluída!", fg="green")
        else:
            self.label_status.config(text="Status: Modelo não carregado ou imagem não processada.", fg="red")

    def exibir_resultado(self, resultado):
        self.canvas.create_text(250, 450, text=resultado, fill="red", font=("Arial", 16))

    def exibir_imagem(self, imagem, texto="Imagem"):
        imagem_pil = Image.fromarray(imagem)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        self.canvas.delete("all")
        self.canvas.create_image(250, 250, image=imagem_tk)
        self.canvas.create_text(250, 450, text=texto, fill="black", font=("Arial", 12))
        self.canvas.image = imagem_tk
        self.janela_principal.update()

    def carregar_dataset(self):
        from sklearn.datasets import load_sample_images
        dataset = load_sample_images()
        imagens = dataset.images
        rotulos = np.array([0, 1])  # Rótulos fictícios
        imagens_processadas = np.array([cv2.resize(img, (224, 224)) / 255.0 for img in imagens])
        return imagens_processadas, rotulos

    def treinar_modelo(self):
        imagens_treinamento, rotulos_treinamento = self.carregar_dataset()
        self.modelo.fit(imagens_treinamento, rotulos_treinamento, epochs=10, batch_size=32)
        self.modelo_carregado = True
        self.label_status.config(text="Status: Modelo treinado com sucesso!", fg="green")
        self.janela_principal.update_idletasks()

janela_principal = tk.Tk()
interface = TratamentoDeImagens(janela_principal)
janela_principal.mainloop()