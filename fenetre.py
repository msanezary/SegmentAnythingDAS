from tkinter import *
from tkinter.messagebox import * # Pour les alertes
from tkinter.filedialog import * # Pour importer les images
from PIL import Image, ImageTk  # On ajoute PIL pour redimensionner l'image

class Application:
     def __init__(self):
          self.image_path = None
          self.photo = None
          self.setup_main_window()
          self.labels = self.charger_labels()

     def charger_labels(self):
          """Charge les labels depuis le fichier labels.txt"""
          try:
              with open('labels.txt', 'r') as file:
                  return [line.strip().strip('"') for line in file.readlines()]
          except FileNotFoundError:
              # Créer le fichier s'il n'existe pas
              with open('labels.txt', 'w') as file:
                  file.write('"label1"\n"label2"\n"label3"')
              return ["label1", "label2", "label3"]

     def sauvegarder_labels(self):
          """Sauvegarde les labels dans le fichier"""
          with open('labels.txt', 'w') as file:
              for label in self.labels:
                  file.write(f"{label}\n")

     def ajouter_nouveau_label(self, fenetre, listbox1):
          """Ajoute un nouveau label aux listes"""
          def valider_nouveau_label():
              nouveau_label = entry_label.get().strip()
              if nouveau_label and nouveau_label not in self.labels:
                  self.labels.append(nouveau_label)
                  listbox1.insert(END, nouveau_label)
                  self.sauvegarder_labels()
                  popup.destroy()
              elif not nouveau_label:
                  showerror("Erreur", "Le label ne peut pas être vide!")
              else:
                  showerror("Erreur", "Ce label existe déjà!")

          popup = Toplevel(fenetre)
          popup.title("Ajouter un label")
          popup.geometry("300x100")
    
          Label(popup, text="Nouveau label:").pack(pady=5)
          entry_label = Entry(popup)
          entry_label.pack(pady=5)
          Button(popup, text="Ajouter", command=valider_nouveau_label).pack(pady=5)
    
     def setup_main_window(self):
          self.fenetre = Tk() # La fenetre principale

          # Créer le frame pour importer l'image
          self.Frame1 = Frame(self.fenetre, borderwidth = 2, relief = GROOVE)
          self.Frame1.pack(side = TOP, padx = 2, pady = 2)
          Label(self.Frame1, text = "Segmentation").pack(padx = 10, pady = 10)

          # Bouton d'import
          self.BoutonImage = Button(self.Frame1, text="Importer une image", 
                                command=self.importer_image)
          self.BoutonImage.pack(side=TOP)

          # Canvas principal
          self.canvas = Canvas(self.Frame1, width=500, height=400, bg="yellow")
          self.canvas.pack()

          # Frame des bouttons pour choisir les types de segmentation
          self.Frame2 = Frame(self.fenetre, borderwidth = 2, relief = GROOVE)
          self.Frame2.pack(side = BOTTOM, padx = 2, pady = 2)
          Label(self.Frame2, text = "Choisir le type de segmentation ...").pack()

          # Les boutons pour segmentation
          self.BoutonAuto = Button(self.Frame2, text = "Automatique", command = self.Auto)
          self.BoutonSemi = Button(self.Frame2, text = "Semi Automatique", command = self.Semi)
          self.BoutonAuto.pack(side = LEFT, padx = 10, pady = 10)
          self.BoutonSemi.pack(side = RIGHT, padx = 10, pady = 10)
     
     def importer_image(self):
          # Obtenir le lien des images
          filepath = askopenfilename(title = "Ouvrir une image", 
                                     filetypes = [('png files', '.png'), ('all files', '.*')])
          
          if filepath:
               # Stocker les images
               self.image_path = filepath
               self.afficher_image(self.canvas)
               self.BoutonImage.config(text = "Choisir une autre image")
     
     def afficher_image(self, canvas_cible):
          # canvas_cible pour choisir la canvas où on veut ouvrir l'image
          if self.image_path:
               image_pil = Image.open(self.image_path)

               # Obtenir les dimensions du canvas
               canvas_width = canvas_cible.winfo_width()
               canvas_height = canvas_cible.winfo_height()

               # Redimensionner l'images
               image_resized = image_pil.resize((canvas_width, canvas_height), 
                                                            Image.Resampling.LANCZOS)
               
               # Créer une nouvelle PhotoImage pour ce canvas
               photo = ImageTk.PhotoImage(image_resized)

               # Garder une référence
               canvas_cible.photo = photo

               # Afficher l'image
               canvas_cible.delete('all')
               canvas_cible.create_image(0, 0, anchor = NW, image = photo)
     
     def verifier_image(self):
          if not self.image_path:
               showwarning('Attention!', 'Vous devez importer une image!')
               return False
          return True
     
     def Auto(self):
        if self.verifier_image():
            fenetreAuto = Toplevel(self.fenetre)
            fenetreAuto.title("Segmentation Automatique")
            
            # Frame principal
            main_frame = Frame(fenetreAuto)
            main_frame.pack(expand=True, fill=BOTH, padx=5, pady=5)
            
            # Frame gauche pour l'image
            frame_gauche = Frame(main_frame)
            frame_gauche.pack(side=LEFT, fill=BOTH, expand=True)
            
            canvas1 = Canvas(frame_gauche, width=500, height=400, bg="yellow")
            canvas1.pack(expand=True)
            
            # Frame droit pour les listes
            frame_droit = Frame(main_frame)
            frame_droit.pack(side=RIGHT, fill=Y, padx=5)
            
            # Première liste
            Label(frame_droit, text="Liste 1").pack()
            listbox1 = Listbox(frame_droit, width=30, height=10)
            listbox1.pack(pady=5)
            
            # Remplir les listes
            for label in self.labels:
                listbox1.insert(END, label)
            
            # Bouton pour ajouter un label
            Button(frame_droit, text="Ajouter un label",
                command=lambda: self.ajouter_nouveau_label(fenetreAuto, listbox1)).pack(pady=10)
            
            # Afficher l'image
            try:
                image_pil = Image.open(self.image_path)
                image_resized = image_pil.resize((500, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image_resized)
                canvas1.image = photo
                canvas1.create_image(0, 0, anchor=NW, image=photo)
            except Exception as e:
                showerror("Erreur", f"Erreur lors de l'affichage de l'image : {str(e)}")

     def Semi(self):
        if self.verifier_image():
            fenetreSemi = Toplevel(self.fenetre)
            fenetreSemi.title("Segmentation Semi-Automatique")
            
            # Frame principal
            main_frame = Frame(fenetreSemi)
            main_frame.pack(expand=True, fill=BOTH, padx=5, pady=5)
            
            # Frame gauche pour l'image
            frame_gauche = Frame(main_frame)
            frame_gauche.pack(side=LEFT, fill=BOTH, expand=True)
            
            canvas2 = Canvas(frame_gauche, width=500, height=400, bg="yellow")
            canvas2.pack(expand=True)
            
            # Frame droit pour les listes
            frame_droit = Frame(main_frame)
            frame_droit.pack(side=RIGHT, fill=Y, padx=5)
            
            # Première liste
            Label(frame_droit, text="Liste 1").pack()
            listbox1 = Listbox(frame_droit, width=30, height=10)
            listbox1.pack(pady=5)
            
            
            # Remplir les listes
            for label in self.labels:
                listbox1.insert(END, label)
            
            # Bouton pour ajouter un label
            Button(frame_droit, text="Ajouter un label",
                command=lambda: self.ajouter_nouveau_label(fenetreSemi, listbox1)).pack(pady=10)
            
            # Afficher l'image
            try:
                image_pil = Image.open(self.image_path)
                image_resized = image_pil.resize((500, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image_resized)
                canvas2.image = photo
                canvas2.create_image(0, 0, anchor=NW, image=photo)
            except Exception as e:
                showerror("Erreur", f"Erreur lors de l'affichage de l'image : {str(e)}")

     def run(self):
          self.fenetre.mainloop()


if __name__ == "__main__":
    app = Application()
    app.run()



# Barre de menu (Exemple)
"""
def alert():
    showinfo("alerte", "Bravo!")

menubar = Menu(fenetre)

menu1 = Menu(menubar, tearoff=0)
menu1.add_command(label="Créer", command=alert)
menu1.add_command(label="Editer", command=alert)
menu1.add_separator()
menu1.add_command(label="Quitter", command=fenetre.quit)
menubar.add_cascade(label="Fichier", menu=menu1)

menu2 = Menu(menubar, tearoff=0)
menu2.add_command(label="Couper", command=alert)
menu2.add_command(label="Copier", command=alert)
menu2.add_command(label="Coller", command=alert)
menubar.add_cascade(label="Editer", menu=menu2)

menu3 = Menu(menubar, tearoff=0)
menu3.add_command(label="A propos", command=alert)
menubar.add_cascade(label="Aide", menu=menu3)

fenetre.config(menu=menubar)
"""
