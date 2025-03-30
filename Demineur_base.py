import numpy as np
import pygame
import time

nombre_de_bombe = 99
x = 26
y = 19
taille = (x, y)


taille_slot = 25

                


class Demineur:

    def __init__(self, x, y,nombre_bombe, grille = np.array([]),FORCE = False):
        pygame.init()
        self.taille_menu = taille_slot*3
        self.screen = pygame.display.set_mode((x * taille_slot, y * taille_slot + self.taille_menu))
        self.x = x
        self.y = y
        self.taille_slot = taille_slot
        self.taille = (x, y)
        self.running = True
        self.nombre_bombe = nombre_bombe
        self.fps = 60
        self.timer = pygame.time.Clock()
        self.grille = grille
        self.maj_coloriage = []



         
    def init_jeu(self):
        self.init = FORCE
        self.perdu = False
        self.fini = False
        self.maj = []
        if not FORCE:
            self.grille = np.zeros(self.taille, dtype=complex)
        self.grille_affichage_image = np.zeros(self.taille, dtype=object)
        self.grille_affichage_valeur = 10*np.ones(self.taille)
        self.init_affichage()


        

    def init_affichage(self):

        self.screen.fill((192,192,192))
        self.bouton_menu = pygame.image.load("./images/normal.png")
        self.bouton_menu = pygame.transform.scale(self.bouton_menu,(2*self.taille_slot,2*self.taille_slot))
        self.screen.blit(self.bouton_menu,(self.x*self.taille_slot//2-self.taille_slot,0))


        for nx in range(self.x):
            for ny in range(self.y):
                self.grille_affichage_image[nx, ny] = pygame.image.load("./images/rien.png")
                self.grille_affichage_image[nx, ny] = pygame.transform.scale(self.grille_affichage_image[nx, ny],
                                                                              (self.taille_slot, self.taille_slot))
                self.screen.blit(self.grille_affichage_image[nx, ny], (self.taille_slot * nx, self.taille_menu + self.taille_slot * ny))



        pygame.display.flip()

    def nb2name(self,k):
        if 0 <= k <= 9:
            return str(int(k))
        elif k == -1:
            return "bombe"
        elif k == -2:
            return "perdu"
        elif k == 10:
            return "rien"
        elif k == 11:
            return "drapeau"
        elif k == 12 :
            return "mauvais_drapeau"
        elif k == -10:
            return "rien_inverse"
        elif k == -11 :
            return "drapeau_inverse"
            

    def coloriage(self):
        for slot in self.maj_coloriage:
            x,y,couleur = slot
            x = x*self.taille_slot
            y = y*self.taille_slot + self.taille_menu
            image = pygame.image.load("./images/" + couleur + ".png")
            image = pygame.transform.scale(image,(self.taille_slot,self.taille_slot))
            image.set_alpha(127)
            self.screen.blit(image,(x,y))
            pygame.display.update((x, y, x + self.taille_slot, y + self.taille_slot ))
            
        self.maj_coloriage = []
     

    def update_display(self,grille,transparence = 1,force = False):
        
        if force:
            self.maj = [(x,y) for x in range(self.x) for y in range(self.y)]

        self.update_liste_affichage_image(grille,int(transparence*255))
        for slot in self.maj :
            x,y = slot
            x = x*self.taille_slot
            y = y*self.taille_slot + self.taille_menu
            pygame.display.update((x, y, x + self.taille_slot, y + self.taille_slot ))
            
        self.maj = []


    def update_liste_affichage_image(self,grille ,transparence = 255):

        for slot in self.maj :
            x,y = slot
            self.grille_affichage_image[x,y] = pygame.image.load("./images/" + self.nb2name(grille[x,y]) + ".png")
            self.grille_affichage_image[x, y] = pygame.transform.scale(self.grille_affichage_image[x, y],
                                                                              (self.taille_slot, self.taille_slot))
            self.grille_affichage_image[x,y].set_alpha(transparence)
            self.screen.blit(self.grille_affichage_image[x, y], (self.taille_slot * x, self.taille_menu + self.taille_slot * y))

    def init_grille(self,couple):
        x0 , y0 = couple
        self.init = True
        self.temps = time.time()
        if self.grille.shape != (self.x,self.y):
            self.grille = np.zeros((self.x,self.y), dtype=complex)
            self.grille_affichage_valeur = 10*np.ones((self.x,self.y))


        for bombe in range(self.nombre_bombe):
            x, y = x0, y0
            while abs(x0 - x) <= 1 and abs(y0 - y) <= 1 and self.grille[x,y] != 1:
                x, y = np.random.randint(self.grille.shape[0]), np.random.randint(self.grille.shape[1])
            self.grille[x, y] = 1


    def compteur_bombe(self,x,y,grille = []):
        if len(grille) == 0:
            grille = self.grille
        return np.real(np.sum(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)]) - grille[x,y])

    def compteur_drapeau(self,slot,grille = []):
        x,y = slot
        if len(grille) == 0:
            grille = self.grille
        return np.imag(np.sum(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)]) - grille[x,y])
    

    def place_drapeau(self,slot,grille,reel = True):

        if grille[slot] == 10:
            if reel :
                grille[slot] = 11
            else :
                grille[slot] = -11
            
            self.maj.append(slot)
            return grille,True

        return grille,False

    def compteur_drapeau_client(self,slot,grille = None,reel = True):
        x,y = slot
        if grille is None :
            grille = self.grille_affichage_valeur

        return np.count_nonzero(np.abs(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)]) == 11)

    
    def compteur_rien(self,slot,grille):
        x,y = slot
        return np.count_nonzero(np.abs(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)]) == 10)

    def ouverture_gauche(self,couple, grille, reel = True, edited = []):

        x,y = couple
        if (x,y) not in self.maj and 0 <= x < self.x and 0 <= y < self.y and np.abs(grille[x,y]) == 10:
            self.maj.append((x,y))

            if reel and self.grille[x,y] == 1 :
                self.perdu = True
                self.perdre_maj()
                grille[x,y] = -2
                self.update_display(grille)
                

            elif reel: #recurence
                case = self.compteur_bombe(x,y)

                if case == 0 :
                    grille[x,y] = case
                    for vecteur in [(i,j) for i in [-1,0,1] for j in [-1,0,1]]:
                        grille,_ = self.ouverture_gauche((x+vecteur[0],y+vecteur[1]),grille)
                        
                else :
                    grille[x,y] = case
            
            else :
                grille[x,y] = -10
                edited.append((x,y))
                
        return grille,edited

    def compteur_revele(self,grille):
        return np.count_nonzero(grille != 10)

    def ouverture_droite(self,slot,grille, reel = True, edited = []):

        x,y = slot
        if 0 <= x < self.x and 0 <= y < self.y:
            case = self.compteur_drapeau_client((x,y),grille,reel)
            if case == grille[x,y]:
                for vecteur in [(i,j) for i in [-1,0,1] for j in [-1,0,1]]:
                    grille,edited = self.ouverture_gauche((x+vecteur[0],y+vecteur[1]),grille,reel,edited)



        return grille,edited        

    def perdre_maj(self):

        for nx,x in enumerate(self.grille):
            for ny,y in enumerate(x):

                if y == 1:
                    self.grille_affichage_valeur[nx,ny] = -1
                    self.maj.append((nx,ny))

                elif y == 1j:
                    self.grille_affichage_valeur[nx,ny] = 12
                    self.maj.append((nx,ny))
        
        self.bouton_menu = pygame.image.load("./images/perdu_smiley.png")
        self.bouton_menu = pygame.transform.scale(self.bouton_menu,(2*self.taille_slot,2*self.taille_slot))
        self.screen.blit(self.bouton_menu,(self.x*self.taille_slot//2-self.taille_slot,0))


    def gagner(self):
        if 10 not in self.grille_affichage_valeur and 1 not in self.grille and 1j not in self.grille and not self.fini :
            self.temps = time.time() - self.temps
            self.fini = True
            self.bouton_menu = pygame.image.load("./images/gagne.png")
            self.bouton_menu = pygame.transform.scale(self.bouton_menu,(2*self.taille_slot,2*self.taille_slot))
            self.screen.blit(self.bouton_menu,(self.x*self.taille_slot//2-self.taille_slot,0))
            pygame.display.update((self.x*self.taille_slot//2-self.taille_slot,0,self.x*self.taille_slot//2+self.taille_slot,2*self.taille_slot))
            print(self.temps)

    def coup_droit(self,x,y,grille_affichage, reel = True):

        if grille_affichage[x,y] == 10: 
            if reel:
                grille_affichage[x,y] = 11
                self.grille[x,y] = self.grille[x,y] + 1j
                self.maj.append((x,y))

            else :
                grille_affichage[x,y] = 101
                self.maj.append((x,y))
            return grille_affichage,True

        elif grille_affichage[x,y] == 11:
            grille_affichage[x,y] = 10
            if reel :
                self.grille[x,y] = self.grille[x,y] - 1j
                self.maj.append((x,y))
            return grille_affichage,True

        elif grille_affichage[x,y] in range(1,9)  :
            grille_affichage,_ = self.ouverture_droite((x,y),grille_affichage,reel)
            return grille_affichage,False
    
        else :
            return grille_affichage,False


    def f(self,G, grille, *args):
        retour = G(*args)
        self.update_display(grille)
        return retour
    
    def animation(self,slot,couleur):
        self.maj.append(slot)
        self.maj_coloriage.append((*slot,couleur))

    def Partie(self):

        self.init_jeu()
        
        p = 0
        while self.running:
            p+=1
            #self.timer.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_button = pygame.mouse.get_pressed()

                    if mouse_button[0]:                          #gauche
                        x, y = pygame.mouse.get_pos()
                        print("g",x,y)

                        if y >= self.taille_menu : 
                            x,y = x//self.taille_slot, (y - self.taille_menu)//self.taille_slot
                            
                        
                            if not self.init :                           #Initialisation de la grille
                                self.init_grille((x,y))
                            self.grille_affichage_valeur,_ = self.ouverture_gauche((x,y),self.grille_affichage_valeur)

                        elif abs(x - self.x*self.taille_slot//2) <= self.taille_slot and abs(y - self.taille_slot) <= self.taille_slot:
                            print("restart")
                            self.init_jeu()
                        
                        
                        


                    elif mouse_button[2] and self.init :                           #droite
                        x, y  = pygame.mouse.get_pos()
                        print("d",x,y)
                        if y >= self.taille_menu:
                            x,y = x//self.taille_slot, (y - self.taille_menu)//self.taille_slot

                            self.grille_affichage_valeur,_ = self.coup_droit(x,y,self.grille_affichage_valeur) 


            self.update_display(self.grille_affichage_valeur)
                   
                                

            if event.type == pygame.QUIT:
                pygame.quit()

            

            """
            if self.bot_init[0]:
                self.coloriage()
                """
            self.gagner()
            time.sleep(0)




    def test_leave(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()




    def get_grille(self):
        return self.grille_affichage_valeur

    def get_cheat(self):
        return self.grille
        
        



GRILLE = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
        1., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 1., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 1., 0.],
       [1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
        1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 1., 1.],
       [0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        1., 0., 0.],
       [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        0., 1., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        0., 0., 0.],
       [1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0.],
       [0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0.,
        0., 0., 0.]])


FORCE = True


if __name__ == "__main__":
        if FORCE:
            fenetre = Demineur(x, y, nombre_de_bombe,GRILLE,FORCE)
        else :
            fenetre = Demineur(x, y, nombre_de_bombe,FORCE=FORCE)

        fenetre.Partie()


#,grille,(19,15)





