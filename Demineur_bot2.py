import numpy as np
import pygame
import time
import itertools

nombre_de_bombe = 99
x = 26
y = 19
taille = (x, y)
grille = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
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
        if self.entre[0]:
            self.init = True
            self.grille_affichage_valeur = self.ouverture_gauche(self.entre[1],self.grille_affichage_valeur)
        

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
        elif k == 100:
            return "rien_inverse"
        elif k == 101 :
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
     

    def update_display(self,grille,transparence = 1,):
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
    
    def compteur_drapeau_client(self,slot,grille = [],reel = True):
        x,y = slot
        if reel:
            grille = self.grille_affichage_valeur
            return np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 11) 

        else :
            return np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 11) + np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 101)

    
    def compteur_rien(self,slot,grille):
        x,y = slot
        return np.count_nonzero(grille[max(x-1,0):min(x+2,self.x),max(y-1,0):min(y+2,self.y)] == 10)

    def ouverture_gauche(self,couple, grille ,reel = True):

        x,y = couple
        if (x,y) not in self.maj and 0 <= x < self.x and 0 <= y < self.y and grille[x,y] == 10:
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
                        grille = self.ouverture_gauche((x+vecteur[0],y+vecteur[1]),grille,reel)
                        
                else :
                    grille[x,y] = case
            
            else :
                grille[x,y] = 100
        return grille

    def compteur_revele(self,grille):
        return np.count_nonzero(grille != 10)

    def ouverture_droite(self,x,y,grille,reel):

        if 0 <= x < self.x and 0 <= y < self.y:
            case = self.compteur_drapeau_client((x,y),grille,reel)
            if case == grille[x,y]:
                for vecteur in [(i,j) for i in [-1,0,1] for j in [-1,0,1]]:
                    grille = self.ouverture_gauche((x+vecteur[0],y+vecteur[1]),grille,reel)



        return grille         

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
            return grille_affichage

        elif grille_affichage[x,y] == 11:
            grille_affichage[x,y] = 10
            if reel :
                self.grille[x,y] = self.grille[x,y] - 1j
                self.maj.append((x,y))
            return grille_affichage

        elif grille_affichage[x,y] in range(1,9)  :
            grille_affichage = self.ouverture_droite(x,y,grille_affichage,reel)
            return grille_affichage
    
        else :
            return grille_affichage


    def f(self,G, grille, *args):
        retour = G(*args)
        self.update_display(grille)
        return retour
    
    def animation(self,slot,couleur):
        self.maj.append(slot)
        self.maj_coloriage.append((slot[0],slot[1],couleur))

    def Partie(self):
        self.init_jeu()
        
        p = 0
        while self.running:
            p+=1
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
                                self.grille_affichage_valeur = self.ouverture_gauche((x,y),self.grille_affichage_valeur)
                            elif abs(x - self.x*self.taille_slot//2) <= self.taille_slot and abs(y - self.taille_slot) <= self.taille_slot:
                                print("restart")
                                self.init_jeu()
                            
                            
                            


                        elif mouse_button[2] and self.init :                           #droite
                            x, y  = pygame.mouse.get_pos()
                            if y >= self.taille_menu:
                                x,y = x//self.taille_slot, (y - self.taille_menu)//self.taille_slot

                                self.grille_affichage_valeur = self.coup_droit(x,y,self.grille_affichage_valeur) 


            if self.bot_init[0] :
                if self.bot_init[1]:
                    self.bot = Bot(self,self.x,self.y,self.nombre_bombe,self.entre[1])
                    self.bot_init[1] = False
                    print('initialisaion bot finie')
                    print(self.bot.etat)

                for q in range(2):
                    if self.bot.etat[q][0]:
                        if self.bot.etat[q][1] :
                            k = 0
                            c = (0,0,0,[0,0])
                            self.bot.etat[q][1] = False
                            slot = self.bot.coup[-1]


                if not self.bot.etat[3][0] :
                    self.bot.grille_completion,self.grille_affichage_valeur,self.bot.coup,self.bot.etat,c,k,slot = self.bot.joue_en_entier(self.bot.grille_completion,self.grille_affichage_valeur,self.bot.coup,self.bot.etat,c,k,slot)
                    self.update_display(self.grille_affichage_valeur)
                
                  
                    
                    
                #Raisonnement plus technique     
                else :
                    if self.bot.etat[3][1] :
                        k = 0
                        c = (0,0,0,[0,0])
                        self.bot.etat[3][1] = False
                        max = self.compteur_revele(self.bot.grille_completion) - self.bot.compteur_fini(self.bot.grille_completion)

                    elif self.bot.etat[3][2][0] :
                        c,k,self.bot.etat[3][2][0],self.bot.etat[3][2][1] = self.bot.interet_slot(c,k,max)
                        if not self.bot.etat[3][2][0] :
                            self.bot.tri_slot.sort(key = lambda x : x[1])
                            self.bot.etat[3][2][2] = True
                            self.bot.init_sur_ensemble()      

                    elif self.bot.etat[3][2][2] :
                        if not self.bot.fin_evidence[self.bot.i_n] :
                            slot = self.bot.init_dans_ensemble()
                            c = (0,0,0,[0,0])
                            k = 0
                        c,k,slot = self.bot.raisonement_secondaire(c,k,slot)
                        

            

                

                    
                            


                        




                    
                                        
                                

            if event.type == pygame.QUIT:
                pygame.quit()

            

            """
            if self.bot_init[0]:
                self.coloriage()
                """
            self.gagner()
            time.sleep(0)












class Bot:

    def __init__(self,fenetre,x,y,nombre_bombe,entre):
        self.x = x
        self.y = y
        self.i_grille_valeur = []
        self.i_grille_completion = []
        self.grille_completion = np.zeros((self.x,self.y), dtype=bool)
        self.coup = [entre]
        self.i_coup = []
        self.nombre_bombe = nombre_bombe
        self.fenetre = fenetre
        self.fenetre.maj_coloriage = []
        self.fenetre.grille_coloriage = np.zeros((self.x,self.y))
        self.etat = [[True,True],[False,True],False,[False,True,[True,False,False,False]]]
        self.i_etat = []
        self.tri_slot = []
        self.fin_evidence = []
        self.ensemble = []
        self.i_n = 0
        self.cases_vides = []
        self.avancement_grille_imaginaire = []

    def factorielle(self,n):
        if n == 0:
            return 1
        else :
            return n*self.factorielle(n-1)
    
    def combinaison(self,n,p):
        return self.factorielle(n)/(self.factorielle(p)*self.factorielle(n-p))

    def partitions(self, N, p):
        # Assurer que les valeurs de N et p sont valides
        p = int(p)
        if p > N or p < 0:
            return []

        # Générer l'ensemble de base [1, N]
        base_set = list(range(1, N + 1))

        # Trouver toutes les combinaisons de p éléments dans base_set
        comb = itertools.combinations(base_set, p)

        partitions = []
        for c in comb:
            subset1 = set(c)
            subset2 = set(base_set) - subset1
            partitions.append((subset1, subset2))

        return partitions

    def print_i_case(self,slot,t,couleur,reel,*arg):
        self.print_case(slot,t,couleur,not reel,*arg)

    def print_case(self,slot,t,couleur,a = False,*arg):
        if a :
            self.fenetre.animation(slot,couleur)
            self.fenetre.coloriage()
            #print(slot,*arg)
            time.sleep(t)

    def compteur_fini(self,grille = []):
        if len(grille) == 0:
            grille = self.grille_completion
        return np.count_nonzero(grille)

    def tourne_en_rond(self,c = (0,0,0,[0,0])):
        n, kx, ky, etat = c
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
        
        return (n,kx,ky,etat)

    def case_finissable(self,slot,grille = []):
        if len(grille) == 0:
            grille = self.fenetre.grille_affichage_valeur
        x,y = slot
        a = grille[x,y] == self.fenetre.compteur_drapeau_client(slot,grille) + self.fenetre.compteur_rien(slot,grille)
        return a
    

    def case_completable(self,slot,grille = []):
        if len(grille) == 0:
            grille = self.fenetre.grille_affichage_valeur
        x,y = slot
        return grille[x,y] == self.fenetre.compteur_drapeau_client(slot,grille)


    def maj_completion(self,c,k,grille_completion,grille_affichage,etat,slot = (),reel = True):

        _,kx,ky,_ = c
        if reel:
            x, y = self.coup[-1]
        else :
            x, y = slot

        x = x + kx
        y = y + ky
        if 0 <= x < self.x and 0 <= y < self.y and grille_affichage[x,y] != 10 :
            if not grille_completion[x,y]  and grille_affichage[x,y] == self.fenetre.compteur_drapeau_client((x,y),grille_affichage,reel) and self.fenetre.compteur_rien((x,y),grille_affichage) == 0:
                grille_completion[x,y] = True
                self.print_i_case((x,y),0,"vert",reel,"salam")
                self.print_case((x,y),0,"vert")
            else :
                if grille_completion[x,y] :
                    self.print_i_case((x,y),0,"vert",reel)
                    self.print_case((x,y),0,"vert")
                else :
                    self.print_i_case((x,y),0,"rouge",reel)
                    self.print_case((x,y),0,"rouge")
            k += 1
            
        
        if k >= self.fenetre.compteur_revele(grille_affichage) : #On fait pas de travail inutile
            etat[0][0] = False
            etat[1][0] = True
            etat[0][1] = True
            time.sleep(0)
            self.fenetre.update_display(grille_affichage)
        else: 
            c = self.tourne_en_rond(c)
        
        return c,k,grille_completion,grille_affichage,etat





    def recherche_case_finissable(self,c,k,grille_completion, grille_affichage, coup, etat, reel = True):


        _,kx,ky,_ = c
        x, y = coup[-1]
        x = x + kx
        y = y + ky
        if 0 <= x < self.x and 0 <= y < self.y and not grille_completion[x,y] and grille_affichage[x,y] != 10:
            cdt = self.case_finissable((x,y),grille_affichage)
            k += 1
            if cdt :
                grille_completion[x,y] = True
                if reel:
                    etat[1][0] = False
                    etat[1][1] = True
                    etat[2] = True
                    self.print_case((x,y),0,"vert")
                    c = self.tourne_en_rond(c)
                    self.fenetre.update_display(grille_affichage)
                
            else :
                cdt = self.case_completable((x,y),grille_affichage)
                if cdt :
                    grille_completion[x,y] = True
                    if reel:
                        etat[1][0] = False
                        etat[1][1] = True
                        etat[0][0] = True
                        self.print_case((x,y),0,"vert")
                        grille_affichage = self.fenetre.coup_droit(x,y,grille_affichage,reel)
                        coup.append((x,y))
                        self.fenetre.update_display(grille_affichage)


                self.print_case((x,y),0,"rouge")
                c = self.tourne_en_rond(c)
                
                if k >= self.fenetre.compteur_revele(grille_affichage) - self.compteur_fini(grille_completion) :
                    if reel :
                        etat[1][0] = False
                        etat[3][0] = True
                        etat[1][1] = True
                        time.sleep(0)
                        print("case finissable pas trouvee")
                        self.fenetre.update_display(grille_affichage)

        else :
            c = self.tourne_en_rond(c)

        return c,k,grille_completion,grille_affichage,(x,y),etat   



    def jouer(self,slot,grille_affichage,grille_completion,etat,coup,reel = True):

        x,y = slot

        for kx in [-1,0,1]:
            for ky in [-1,0,1]:
                if 0 <= x + kx < self.x and 0 <= y + ky < self.y and not grille_completion[x + kx,y + ky] :
                    if grille_affichage[x + kx, y + ky] == 10:
                        grille_completion[x + kx, y + ky] = True
                        grille_affichage = self.fenetre.coup_droit(x + kx, y + ky,grille_affichage,reel)
                        coup.append((x + kx, y + ky))
                        self.print_case((x + kx, y + ky),0,"vert")
                    else :
                        self.print_case((x + kx, y + ky),0,"rouge")


        self.fenetre.update_display(grille_affichage)
        for kx in [-1,0,1]:
            for ky in [-1,0,1]:
                if 0 <= x + kx < self.x and 0 <= y + ky < self.y and not grille_completion[x + kx,y + ky] and grille_affichage[x + kx, y + ky] not in [0,10,11]:
                    grille_affichage = self.fenetre.coup_droit(x + kx, y + ky,grille_affichage,reel)
                    self.print_case((x + kx, y + ky),0.1,"bleu")
        self.fenetre.update_display(grille_affichage)
        etat[2] = False
        etat[0][0] = True

        return grille_affichage,grille_completion,etat,coup
        

    def interet_slot(self,c,k,max):
        x,y = self.coup[-1]
        _,kx,ky,_ = c
        x = x + kx
        y = y + ky

        if k < max:
            if 0 <= x < self.x and 0 <= y < self.y and not self.grille_completion[x,y] and self.fenetre.grille_affichage_valeur[x,y] != 10:
                k += 1
                combis = self.combinaison(self.fenetre.compteur_rien((x,y),self.fenetre.grille_affichage_valeur),self.fenetre.grille_affichage_valeur[x,y] - self.fenetre.compteur_drapeau_client((x,y)))
                assert(combis >= 2)
                if combis == 2:  
                    self.tri_slot.append([x,y,combis])
                    c = self.tourne_en_rond(c)
                    return c,k,False,False
                else :
                    self.tri_slot.append([x,y,combis])
            c = self.tourne_en_rond(c)
            return c,k,True,False
        else :
            c = self.tourne_en_rond(c)
            return c,k,True,True

    def liste_case_vide(self,slot,grille = []):
        if len(grille) == 0:
            grille = self.fenetre.grille_affichage_valeur
        x,y = slot
        L = []
        for kx in [-1,0,1]:
            for ky in [-1,0,1]:
                if 0 <= x + kx < self.x and 0 <= y + ky < self.y and grille[x + kx, y + ky] == 10:
                    L.append((x + kx, y + ky))
        return np.array(L)

    def intersection(self,grilles):
        for k in range(len(grilles)):
            grilles[k] = [grilles[k] == 101,grilles[k] == 100]
        
        for k in range(len(grilles)):
            grilles[0][0] = grilles[0][0] & grilles[k][0]
            grilles[0][1] = grilles[0][1] & grilles[k][1]

        return grille[0][0],grille[0][1]

    def joue_en_entier(self,grille_completion,grille_affichage,coup,etat,c,k,slot,reel = True):
        if etat[0][0] :
            if not reel:
                print("Completion",k)
                time.sleep(0)
            c,k,grille_completion,grille_affichage,etat = self.maj_completion(c,k,grille_completion,grille_affichage,etat,slot,reel)
            
        
        #recherche de case finissable
        elif etat[1][0]:


            c,k,grille_completion,grille_affichage,slot,etat = self.recherche_case_finissable(c,k,grille_completion,grille_affichage,coup,etat,reel)
        
        
        #On joue littéralement
        elif etat[2]: 
            if not reel:
                print("Joue")
                time.sleep(0.5)
            grille_affichage,grille_completion,etat,coup = self.jouer(slot,grille_affichage,grille_completion,etat,coup,reel)
            
        return grille_completion,grille_affichage,coup,etat,c,k,slot


    def init_sur_ensemble(self):                  #On initialise sur l'ensemble choisi
        self.ensemble = []
        self.i_n = -1
        self.fin_evidence = [False]

    def init_dans_ensemble(self):                      #On initialise toutes les grilles qui sont internes a l'ensemble
        self.fin_evidence[-1] = True 
                                                                            #Propre a l'ensemble
        self.avancement_grille_imaginaire = []              #Pour chacune des grilles de l'enemble
        if len(self.tri_slot) > 0:
            x,y,_ = self.tri_slot.pop(0)
            
            nb_drapeau_manquants = self.fenetre.grille_affichage_valeur[x,y] - self.fenetre.compteur_drapeau_client((x,y))
            
            self.ensemble = self.partitions(self.fenetre.compteur_rien((x,y),self.fenetre.grille_affichage_valeur),nb_drapeau_manquants)
            self.cases_vides = self.liste_case_vide((x,y))

            for p in self.ensemble:
                self.fin_evidence.append(False)
                self.i_grille_valeur.append(np.array(np.copy(self.fenetre.grille_affichage_valeur)))
                self.i_grille_completion.append(np.array(np.copy(self.grille_completion)))
                
                for k in range(2):
                    for q in p[k]:
                        if k < 1:
                            
                            self.i_grille_valeur[-1][self.cases_vides[q-1][0],self.cases_vides[q-1][1]] = 101
                            
                            
                        else :
                            self.i_grille_valeur[-1][self.cases_vides[q-1][0],self.cases_vides[q-1][1]] = 100

                        self.i_grille_completion[-1][self.cases_vides[q-1][0],self.cases_vides[q-1][1]] = True

                self.i_coup.append(self.coup)
                self.i_etat.append([[True,True],[False,True],False])


                
                self.avancement_grille_imaginaire.append(False)
            

        else:
            print("Y'a pu de slot a traiter") #On est passé ici une seconde fois alors qu'on devrait pas pour l'instant

        return self.cases_vides[0]


    def affiche_hypotheses(self):
        for k in range(2):
            for q in self.ensemble[self.i_n][k] :
                self.fenetre.maj.append(self.cases_vides[q-1])
        self.fenetre.update_display(self.i_grille_valeur[-1],1/2)

    def raisonement_secondaire(self,c,k,slot):
        while (not self.fin_evidence[self.i_n] and self.i_n < len(self.ensemble)) or self.i_n == -1:
            self.i_n += 1
            print("On passe a la grille suivante")
            self.affiche_hypotheses()
        
        if self.i_n < len(self.ensemble):

            for q in range(2):
                if self.i_etat[self.i_n][q][0]:
                    if self.i_etat[self.i_n][q][1] :
                        k = 0
                        c = (0,0,0,[0,0])
                        self.i_etat[self.i_n][q][1] = False
                        slot = self.cases_vides[self.i_n]

            self.i_grille_completion[self.i_n],self.i_grille_valeur[self.i_n],self.i_coup[self.i_n],self.i_etat[self.i_n],c,k,slot = self.joue_en_entier(self.i_grille_completion[self.i_n],self.i_grille_valeur[self.i_n],self.i_coup[self.i_n],self.i_etat[self.i_n],c,k,slot,False)

             #Probleme : La completion semble imparfaite, en particulier la ou on pose le drapeau imaginaire





        else :
            print("On a fini de raisonner sur l'ensemble")


        return c,k,slot
        

        

        
        #Il faut regarder tout ce que ca implique pour chacune des grilles imaginaires...
        


        



if __name__ == "__main__":
        fenetre = Fenetre(x, y, taille_slot, nombre_de_bombe,True,grille,(0,0))
        fenetre.Partie()      


#,grille,(19,15)





