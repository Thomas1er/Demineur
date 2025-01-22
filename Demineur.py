import numpy as np
import pygame
import time 

nombre_de_bombe = 99
x = 26
y = 19
taille = (x, y)
grille = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0.]])

taille_slot = 25

                


class Fenetre:

    def __init__(self, x, y, taille_slot,nombre_bombe, bot, grille = np.array([]), entre = ()):
        pygame.init()
        self.taille_menu = taille_slot*3
        self.screen = pygame.display.set_mode((x * taille_slot, y * taille_slot + self.taille_menu))
        self.x = x
        self.y = y
        self.taille_slot = taille_slot
        self.taille = (x, y)
        self.running = True
        self.nombre_bombe = nombre_bombe
        self.bot_init = [bot,bot]
        self.fps = 60
        self.timer = pygame.time.Clock()
        self.grille = grille
        if entre != ():
            self.entre = [True,entre]
        else :
            self.entre = [False,()]



         
    def init_jeu(self):
        self.init = False
        self.perdu = False
        self.fini = False
        self.maj = []
        if self.grille.shape == np.array([]).shape:
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

    def update_display(self):
        self.update_liste_affichage_image()
        for slot in self.maj :
            x,y = slot
            x = x*self.taille_slot
            y = y*self.taille_slot + self.taille_menu
            pygame.display.update((x, y, x + self.taille_slot, y + self.taille_slot ))
            
        self.maj = []


    def update_liste_affichage_image(self):
        for slot in self.maj :
            x,y = slot
            self.grille_affichage_image[x,y] = pygame.image.load("./images/" + self.nb2name(self.grille_affichage_valeur[x,y]) + ".png")
            self.grille_affichage_image[x, y] = pygame.transform.scale(self.grille_affichage_image[x, y],
                                                                              (self.taille_slot, self.taille_slot))
            self.screen.blit(self.grille_affichage_image[x, y], (self.taille_slot * x, self.taille_menu + self.taille_slot * y))

    def init_grille(self,x0,y0):
        self.init = True
        self.temps = time.time()
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
    
    def compteur_drapeau_client(self,slot,grille = []):
        x,y = slot
        if len(grille) == 0:
            grille = self.grille_affichage_valeur
        return np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 11)
    
    def compteur_rien(self,slot,grille = []):
        x,y = slot
        if len(grille) == 0:
            grille = self.grille_affichage_valeur
        return np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 10)

    def ouverture_gauche(self,x,y):

        if (x,y) not in self.maj and 0 <= x < self.x and 0 <= y < self.y and self.grille_affichage_valeur[x,y] == 10:
            self.maj.append((x,y))

            if self.grille[x,y] == 1:
                self.perdu = True
                self.perdre_maj()
                self.grille_affichage_valeur[x,y] = -2
                self.update_display()
                

            else :
                case = self.compteur_bombe(x,y)

                if case == 0 :
                    self.grille_affichage_valeur[x,y] = case
                    for vecteur in [(i,j) for i in [-1,0,1] for j in [-1,0,1]]:
                        self.ouverture_gauche(x+vecteur[0],y+vecteur[1])
                        
                else :
                    self.grille_affichage_valeur[x,y] = case

    def ouverture_droite(self,x,y):

        if 0 <= x < self.x and 0 <= y < self.y:
            case = self.compteur_drapeau_client((x,y))
            if case == self.grille_affichage_valeur[x,y]:
                for vecteur in [(i,j) for i in [-1,0,1] for j in [-1,0,1]]:
                    self.ouverture_gauche(x+vecteur[0],y+vecteur[1])              

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

    def coup_droit(self,x,y):

        if self.grille_affichage_valeur[x,y] == 10:
            self.grille_affichage_valeur[x,y] = 11
            self.grille[x,y] = self.grille[x,y] + 1j
            self.maj.append((x,y))

        elif self.grille_affichage_valeur[x,y] == 11:
            self.grille_affichage_valeur[x,y] = 10
            self.grille[x,y] = self.grille[x,y] - 1j
            self.maj.append((x,y))

        elif self.grille_affichage_valeur[x,y] in range(1,9) :
            self.ouverture_droite(x,y)

    def f(self,G, *args):
        retour = G(*args)
        self.update_display()
        return retour
    
    def animation(self,slot,couleur):
        self.maj.append(slot)
        self.maj_coloriage.append((slot[0],slot[1],couleur))

    def Partie(self):
        self.init_jeu()
        k = 0
        while self.running:
            k+=1
            #self.timer.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.bot_init[0] :
                        mouse_button = pygame.mouse.get_pressed()


                        if mouse_button[0]:                             #gauche
                            x, y = pygame.mouse.get_pos()
                            if y >= self.taille_menu : 
                                x,y = x//self.taille_slot, (y - self.taille_menu)//self.taille_slot
                                
                            
                                if not self.init :                           #Initialisation de la grille
                                    self.init_grille(x,y)
                                self.ouverture_gauche(x,y)
                            elif abs(x - self.x*self.taille_slot//2) <= self.taille_slot and abs(y - self.taille_slot) <= self.taille_slot:
                                print("restart")
                                self.init_jeu()
                            
                            
                            


                        elif mouse_button[2] and self.init :                           #droite
                            x, y  = pygame.mouse.get_pos()
                            if y >= self.taille_menu:
                                x,y = x//self.taille_slot, (y - self.taille_menu)//self.taille_slot

                                self.coup_droit(x,y)
                                    
                                

                if event.type == pygame.QUIT:
                    pygame.quit()

            if self.bot_init[0] :
                if self.bot_init[1]:
                    self.bot = Bot(self,self.x,self.y,self.nombre_bombe)
                    self.bot_init[1] = False
                    self.f(self.bot.initialisation)
                    self.f(self.bot.maj_completion,self.bot.coup[-1][:-1],self.grille_affichage_valeur,self.bot.grille_completion)

                self.bot.case_incomplete = []
                slot = self.bot.coup[-1][:-1]
                slot, existe_case_incomplete = self.f(self.bot.trouve_case_incomplete,self.grille_affichage_valeur,self.bot.grille_completion,slot)
                self.bot.case_incomplete.append(slot)

                if existe_case_incomplete :

                    slot,succes = self.f(self.bot.recherche_reguliere,slot)
 
                    if succes :
                        if self.compteur_drapeau_client(slot) != self.grille_affichage_valeur[slot]:
                            
                        
                            self.bot.drapeau_manquant(slot)
                            
                            
                        else :
                            self.animation(slot,"vert")
                            x,y = slot
                            self.coup_droit(x,y)
                            self.bot.coup.append((x,y,1))
                        self.update_display()


                        time.sleep(0.02)
                        self.f(self.bot.maj_completion,slot,self.grille_affichage_valeur,self.bot.grille_completion,[])
                        self.f(self.bot.print_completion,self.bot.grille_completion,2)

                if not succes :
                    print("Je ne sais pas encore faire")


    


            self.update_display()
            """
            if self.bot_init[0]:
                self.coloriage()
                """
            self.gagner()
            time.sleep(0.1)























class Bot:

    def __init__(self,fenetre,x,y,nombre_bombe):
        self.x = x
        self.y = y
        self.grille_imaginaire = []
        self.grille_completion = np.zeros((self.x,self.y), dtype=bool)
        self.grille_completion_imaginaire = []
        self.coup = []
        self.coup_imaginaire = []
        self.case_incomplete = []
        self.case_incomplete_imaginaire = []
        self.nombre_bombe = nombre_bombe
        self.fenetre = fenetre
        self.fenetre.maj_coloriage = []
        self.fenetre.grille_coloriage = np.zeros((self.x,self.y))

    def initialisation(self):
        if not self.fenetre.entre[0]:
            x,y = np.random.randint(self.x),np.random.randint(self.y)
            self.fenetre.init_grille(x,y)
        else :
            x,y = self.fenetre.entre[1]
        print(x,y)
        print(self.fenetre.grille)
        self.fenetre.ouverture_gauche(x,y)
        self.coup.append((x,y,0))
        self.fenetre.animation((x,y),"vert")

    def drapeau_manquant(self,slot):
        liste_drapeau = self.fenetre.f(self.listage_drapeau,slot,self.fenetre.grille_affichage_valeur)
        self.fenetre.f(self.ouverture_drapeau,liste_drapeau)
        self.fenetre.update_display()

        
    
    def trouve_case_incomplete(self,grille,grille_completion,slot):
        x,y = slot
        kx = 0
        ky = -1
        n = 1
        etat = [1,0]
        a = 0 <= kx + x < self.x and 0 <= ky + y < self.y
        b = n < self.x + self.y
        if a:
            c = grille_completion[kx + x,ky + y]
            d = grille[kx + x,ky + y] in range(1,9)
        else :
            c = True
            d = False
        while  b and (not a or (a and (not d or c and d))):
            a = 0 <= kx + x < self.x and 0 <= ky + y < self.y
            b = n < self.x + self.y
            if a:
                c = grille_completion[kx + x,ky + y]
                d = grille[kx + x,ky + y] in range(1,9)
            else : 
                c = True
                d = False

            if 0 <= kx + x < self.x and 0 <= ky + y < self.y :
                self.fenetre.animation((kx + x,ky + y),"bleu")
                self.fenetre.coloriage()
                time.sleep(0.1)

            n,kx,ky,etat = self.tourne_en_rond(n,kx,ky,etat)
            

        if n >= self.x + self.y :
            return (-1, -1), False
        self.case_incomplete.append((kx + x,ky + y))
        return (kx + x, ky + y), True

    def test_case_incomplete(self,x,y,grille_completion):
        return grille_completion[x,y]


    def maj_completion(self,slot,grille,grille_completion,fait = []): ##A revoir
        x,y = slot
        if not grille_completion[x,y] and slot not in fait:
            fait.append(slot)
            time.sleep(0.01)
            
            if (grille[x,y] == self.fenetre.compteur_drapeau_client((x,y),grille) and self.fenetre.compteur_rien((x,y),grille) == 0) or grille[x,y] == 11:

                grille_completion[x,y] = True
                self.fenetre.animation((x,y),"vert")
                for k in [[i,j] for i in [-1,0,1] for j in [-1,0,1]]:
                    if k != [0,0]:
                        x1,y1 = x + k[0], y + k[1]

                        if 0 <= x1 < self.x and 0 <= y1 < self.y:
                            self.maj_completion((x1,y1),grille,grille_completion,fait)
            else :
                self.fenetre.animation((x,y),"rouge")
        self.fenetre.coloriage()
        return grille_completion

    def listage_drapeau(self,slot,grille):
        x,y = slot
        liste_drapeau = []
        for k in [[i,j] for i in [-1,0,1] for j in [-1,0,1]]:
            if k != [0,0] and 0 <= x + k[0] < self.x and 0 <= y + k[1] < self.y and grille[x + k[0], y + k[1]] == 10:
                self.fenetre.coup_droit(x + k[0], y + k[1])
                self.coup.append((x +k[0],y + k[1],1))
                self.fenetre.animation((x + k[0], y + k[1]),"vert")
                liste_drapeau.append((x + k[0], y + k[1]))
                               
            elif k != [0,0] and 0 <= x + k[0] < self.x and 0 <= y + k[1] < self.y :
                self.fenetre.animation((x + k[0], y + k[1]),"rouge")

            time.sleep(0.01)
            self.fenetre.coloriage()
        if 0 <= x < self.x and 0 <= y < self.y:
            self.fenetre.coup_droit(x,y)
        return liste_drapeau

    def ouverture_drapeau(self,liste_drapeau): ##Probleme, certaines cases ne s'ouvrent pas, bug affichage ?
        fait = []
        
        for slot in liste_drapeau:
            x,y = slot
            for k in [[i,j] for i in [-1,0,1] for j in [-1,0,1]]:
                x1,y1 = x + k[0], y + k[1]

                if 0 <= x1 < self.x and 0 <= y1 < self.y and (x1,y1) not in fait:
                    fait.append((x1,y1))
                    if self.fenetre.grille_affichage_valeur[x1,y1] in range(1,9) :
                        self.fenetre.animation((x1,y1),"vert")
                        self.fenetre.coup_droit(x1,y1)
                        time.sleep(0.5)
                        self.coup.append((x1,y1,1))
                    else :
                        self.fenetre.animation((x1,y1),"rouge")
                        

                    time.sleep(0.01)
                    self.fenetre.coloriage()


    def print_completion(self,grille_completion,t):
        for nx,x in enumerate(grille_completion):
            for ny,y in enumerate(x):
                if y:
                    self.fenetre.animation((nx,ny),"vert")
                else:
                    self.fenetre.animation((nx,ny),"rouge")
        self.fenetre.coloriage()
        time.sleep(t)



    def tourne_en_rond(self,n,kx,ky,etat):
        if ky == -n and etat == [1,0]:
            kx += 1
            if kx == n:
                etat = [0,1]
        elif kx == n and etat == [0,1]:
            ky += 1
            if ky == n:
                etat = [-1,0]
        elif ky == n and etat == [-1,0]:
            kx -= 1
            if kx == -n:
                etat = [0,-1]
        elif kx == -n and etat == [0,-1]:
            ky -= 1
            if ky == -n:
                etat = [0,0]
        elif ky == -n and kx == -n and etat == [0,0]:
                n += 1
                kx = -n+1
                ky = -n
                etat = [1,0]
        
        return n,kx,ky,etat


    def print_case(self,slot,t,*arg):
        self.fenetre.animation(slot,"bleu")
        self.fenetre.coloriage()
        print(slot,*arg)
        time.sleep(t)


    """
    def liste_case_incomplete(self,x0,y0,grille,grille_completion,liste_incomplete):
        if (x0,y0) not in liste_incomplete:
            if not self.test_case_incomplete(x0,y0,grille_completion):
                liste_incomplete.append(x0,y0)

                for k in [[i,j] for i in [-1,0,1] for j in [-1,0,1]] :

                    if k != [0,0]:
                        x,y = x0 + k[0], y0 + k[1]
                        self.liste_case_incomplete(min(max(0, x),self.x), min(max(0, y),self.y),grille,grille_completion,liste_incomplete)
    """

    def ajout_case_incomplete(self,slot,grille,grille_completion,case_incomplete,fait,trouve = False): ##Parfois il ne renvoie pas certaines case qui peuvent etre ouvertent, surement un probleme que fait ou case_incomplete est non nul au depart
        x0,y0 = slot
        n = 1
        kx = 0
        ky = -1
        etat = [1,0]
        while (not trouve) and n < self.x + self.y:
            x,y = x0 + kx, y0 + ky
            if (x,y) not in fait and 0 <= x < self.x and 0 <= y < self.y and grille[x,y] in range(0,9):
                time.sleep(0.01)
                if (not grille_completion[x,y]) and (grille[x,y] != 0) and ((x,y) not in case_incomplete):
                    trouve = True
                    self.fenetre.animation((x,y),"vert")
                    self.fenetre.coloriage()
                    case_incomplete.append((x,y))
                    return (x,y),trouve,fait,case_incomplete
                else :
                    self.fenetre.animation((x,y),"rouge")
                    fait.append((x,y))
                    self.fenetre.coloriage()


            n,kx,ky,etat = self.tourne_en_rond(n,kx,ky,etat)

        return (-1,-1),False,fait,case_incomplete
        

                            


            

                            



    def tri_case_incomplete(self,liste_incomplete,grille):
        return sorted(liste_incomplete, key = lambda grille : grille[liste_incomplete[0],liste_incomplete[1]] - self.fenetre.compteur_drapeau_client((liste_incomplete[0],liste_incomplete[1]),grille))
                
    def finissable(self,grille,slot):
        x,y = slot
        return grille[x,y] == self.fenetre.compteur_drapeau_client((x,y),grille) + self.fenetre.compteur_rien((x,y),grille)
    
    def finition(self,x,y,grille):
        for k in [[i,j] for i in [-1,0,1] for j in [-1,0,1]]:
            if grille[x + k[0], y + k[1]] == 10:
                grille[x + k[0], y + k[1]] = 11
        return grille
    
    def recherche_reguliere(self,slot):
        reussi_a_joue = self.finissable(self.fenetre.grille_affichage_valeur,slot)
        b = True
        self.case_incomplete = []
        fait = []
        while not b or not reussi_a_joue:
            time.sleep(0.01)
            slot,b,fait,self.case_incomplete = self.ajout_case_incomplete(slot,self.fenetre.grille_affichage_valeur,self.grille_completion,self.case_incomplete,fait)     
            reussi_a_joue = self.finissable(self.fenetre.grille_affichage_valeur,slot)

        return slot,reussi_a_joue





    

    
            


if __name__ == "__main__":
        fenetre = Fenetre(x, y, taille_slot, nombre_de_bombe,True)
        fenetre.Partie()      


#,grille,(19,15)

"""
# Exemple d'utilisation
if __name__ == "__main__":
    bot = Bot(x, y, taille_slot, nombre_de_bombe)
    bot.Partie()
"""




