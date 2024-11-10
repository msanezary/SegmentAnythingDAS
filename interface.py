from tkinter import *
from tkinter.messagebox import *  # Pour les alertes
from tkinter.filedialog import *  # Pour importer les images
from PIL import Image, ImageTk  # On ajoute PIL pour redimensionner l'image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        self.fenetre = Tk()  # La fenêtre principale

        # Créer le frame pour importer l'image
        self.Frame1 = Frame(self.fenetre, borderwidth=2, relief=GROOVE)
        self.Frame1.pack(side=TOP, padx=2, pady=2)
        Label(self.Frame1, text="Segmentation").pack(padx=10, pady=10)

        # Bouton d'import
        self.BoutonImage = Button(self.Frame1, text="Importer une image", 
                                  command=self.importer_image)
        self.BoutonImage.pack(side=TOP)

        # Canvas principal
        self.canvas = Canvas(self.Frame1, width=500, height=400, bg="yellow")
        self.canvas.pack()

        # Frame des boutons pour choisir les types de segmentation
        self.Frame2 = Frame(self.fenetre, borderwidth=2, relief=GROOVE)
        self.Frame2.pack(side=BOTTOM, padx=2, pady=2)
        Label(self.Frame2, text="Choisir le type de segmentation ...").pack()

        # Les boutons pour segmentation
        self.BoutonAuto = Button(self.Frame2, text="Automatique", command=self.Auto)
        self.BoutonSemi = Button(self.Frame2, text="Semi Automatique", command=self.Semi)
        self.BoutonAuto.pack(side=LEFT, padx=10, pady=10)
        self.BoutonSemi.pack(side=RIGHT, padx=10, pady=10)

    def importer_image(self):
        # Obtenir le lien des images
        filepath = askopenfilename(title="Ouvrir une image", 
                                   filetypes=[('png files', '.png'), ('all files', '.*')])
        
        if filepath:
            # Stocker les images
            self.image_path = filepath
            self.afficher_image(self.canvas)
            self.BoutonImage.config(text="Choisir une autre image")

    def afficher_image(self, canvas_cible):
        # canvas_cible pour choisir la canvas où on veut ouvrir l'image
        if self.image_path:
            image_pil = Image.open(self.image_path)

            # Obtenir les dimensions du canvas
            canvas_width = canvas_cible.winfo_width()
            canvas_height = canvas_cible.winfo_height()

            # Redimensionner l'image
            image_resized = image_pil.resize((canvas_width, canvas_height), 
                                             Image.Resampling.LANCZOS)
            
            # Créer une nouvelle PhotoImage pour ce canvas
            photo = ImageTk.PhotoImage(image_resized)

            # Garder une référence
            canvas_cible.photo = photo

            # Afficher l'image
            canvas_cible.delete('all')
            canvas_cible.create_image(0, 0, anchor=NW, image=photo)

    def verifier_image(self):
        if not self.image_path:
            showwarning('Attention!', 'Vous devez importer une image!')
            return False
        return True


##################### SECTION AUTO ###########################################""


    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

        ax.imshow(img)

    def init_model(self, device='cuda'):
        sam2_checkpoint = "C:/Users/jamai/projects/SegmentAnythingDAS/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
        # Initialize the model
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        return sam2

    def segment_image_auto(self, image_path, model):
        # Read the image from the path
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
    
        # Generate the mask for the given image
        mask_generator = SAM2AutomaticMaskGenerator(model)
        masks = mask_generator.generate(image)
    
        return masks, image

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

            # Initialize the segmentation model if not already done
            if not hasattr(self, 'model'):
                self.model = self.init_model(device='cuda')  # Or 'cpu' if CUDA is unavailable

            # Generate segmentation masks for the image
            try:
                # Read and segment the image
                masks, original_image = self.segment_image_auto(self.image_path, self.model)

                # Convert segmented output to an ImageTk format for display in Tkinter
                plt.figure(figsize=(5, 4))
                plt.imshow(original_image)
                self.show_anns(masks)
                plt.axis('off')

                # Save and display the resulting segmented image in the canvas
                plt.savefig('segmented_output.png')
                segmented_image = Image.open('segmented_output.png').resize((500, 400), Image.LANCZOS)
                photo_segmented = ImageTk.PhotoImage(segmented_image)
                canvas1.photo = photo_segmented
                canvas1.create_image(0, 0, anchor=NW, image=photo_segmented)

            except Exception as e:
                showerror("Erreur", f"Erreur de segmentation : {e}")




############################ SECTION SEMI-AUTO ##################################################""

    def show_mask(self, mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


    
    def segment_image_semi(self, image_path, model, input_point,input_label):
        # Read the image from the path
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        
        predictor = SAM2ImagePredictor(model)
        predictor.set_image(image)


        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        return image, masks, scores
    
    def on_canvas_click(self, event, canvas, image_pil):
        # Get the position where the user clicked (event.x, event.y)
        click_x = event.x
        click_y = event.y
        
        # Convert canvas coordinates to image coordinates (optional if needed)
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
        image_width, image_height = image_pil.size
        input_point = np.array([[click_x * image_width / canvas_width, click_y * image_height / canvas_height]])

        # Perform segmentation using the clicked point
        input_label = np.array([1])
        image, masks, scores = self.segment_image_semi(self.image_path, self.model, input_point,input_label)

        # Display the segmented image with the mask
        plt.figure(figsize=(5, 4))
        plt.imshow(image)
        self.show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)  # Show masks with clicked point
        plt.axis('off')
        plt.show()

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
            
            # Initialize the segmentation model if not already done
            if not hasattr(self, 'model'):
                self.model = self.init_model(device='cuda')  # Or 'cpu' if CUDA is unavailable

            # Display image on canvas
            try:
                image_pil = Image.open(self.image_path)
                image_resized = image_pil.resize((500, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image_resized)
                canvas2.image = photo
                canvas2.create_image(0, 0, anchor=NW, image=photo)
                
                # Bind the click event on canvas to capture the point
                canvas2.bind("<Button-1>", lambda event: self.on_canvas_click(event, canvas2, image_pil))
                
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
