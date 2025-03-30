import numpy as np 
import Demineur_base as db
import time as t
from collections import deque

ANIMATE = False
K = 0


def fact(n):
    if n <= 0:
        return 1
    else:
        return n*fact(n-1)

def combinaison(n,p):
    if n < 0 or p < 0 or n < p:
        return np.inf
    elif n == p or p == 0:
        return 1
    elif p == 1:
        return n
    elif p == n-1:
        return n
    else:
        return fact(n)//(fact(p)*fact(n-p))


class Grille():
    def __init__(self,X,Y,grille = None, comb = 1, parent = None):
        self.X = X
        self.Y = Y
        self.comb = comb
        if grille is None:
            self.grille = np.zeros((X,Y))
        else:
            self.grille = grille
        self.children = []
        self.parent = parent


    def __add__(self, other):
        if not isinstance(other, Grille):
            raise ValueError("Can only add another Grille object.")
        if self.X != other.X or self.Y != other.Y:
            raise ValueError("Grilles must have the same dimensions.")
        
        result = self.grille.copy()
        mask = (self.grille == 10)
        result[mask] = other.grille[mask]
        return Grille(self.X, self.Y, result)
    
    def __getitem__(self, key):
        return self.grille[key]
    
    def __setitem__(self, key, value):
        self.grille[key] = value
        return self
    
    def __str__(self):
        return self.grille.__str__()
    
    def __eq__(self, other):
        if isinstance(other, Grille):
            return np.array_equal(self.grille, other.grille)
        elif isinstance(other, int):
            return np.all(self.grille == other)
    
    def __ne__(self, other):
        return not np.all(self.grille == other.grille)
    
    def __gt__(self, other):
        vide = (self.grille == 10)
        return (other.grille[vide] != 10).any()
    
    def __lt__(self, other):
        vide = (other.grille == 10)
        return (self.grille[vide] != 10).any()
    
    def __ge__(self, other):
        return not self < other
    
    def __le__(self, other):
        return not self > other
    
    def __len__(self):
        return self.X * self.Y
    
    def __iter__(self):
        return self.grille.__iter__()
    
    
    def __contains__(self, item):
        return item in self.grille
    
    def __reversed__(self):
        return self.grille.__reversed__()

    def copy(self):
        return Grille(self.X, self.Y, self.grille.copy(), self.comb, self.parent)
    
        

class Bot():
    def __init__(self,X,Y,nb_bombe, GRILLE = np.array([]),FORCE = False):
        self.X = X
        self.Y = Y
        self.nb_bombe = nb_bombe
        self.demineur = db.Demineur(X,Y,nb_bombe,GRILLE)
        self.grille_completion = np.zeros((self.X,self.Y))
        self.grille_proba = np.zeros((self.X,self.Y))
        self.taches_finies = []
        self.coups = []


        self.demineur.init_jeu()

        if FORCE:
            x0,y0 = 8,4
        else :
            x0,y0 = self.random_play()
            self.demineur.init_grille((x0,y0))


        self.coups.append((x0,y0))
        self.grille_affichage = Grille(self.X,self.Y,self.demineur.ouverture_gauche((x0,y0),self.demineur.get_grille())[0])



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


    def test_inside(self,x,y):
        return 0 <= x < self.X and 0 <= y < self.Y             

    def random_play(self):
        x,y = np.random.randint(0,self.X),np.random.randint(0,self.Y)
        return x,y

    def compteur_rien(self,slot,grille):
        x,y = slot
        return np.count_nonzero(grille[max(x-1,0):min(x+2,self.X),max(y-1,0):min(y+2,self.Y)] == 10)

    def compteur_drapeau(self,slot,grille):
        x,y = slot
        return np.count_nonzero(np.abs(grille[max(x-1,0):min(x+2,self.X),max(y-1,0):min(y+2,self.Y)]) == 11) 

    def test_completable(self,slot,grille):
        return (grille[slot] == self.compteur_drapeau(slot,grille) and self.compteur_rien(slot,grille) == 0) or np.abs(grille[slot]) == 11 or grille[slot] == -10
    
    def liste_rien(self,slot,grille):
        return [(x,y) for x,y in self.liste_voisins(slot) if np.abs(grille[x,y]) == 10]

    def update_completion(self,grille,completion):


        N = self.X*self.Y
        x0,y0 = self.coups[-1]
        fait = 0
        
        n,kx,ky,etat = 0,0,0,[0,0]
        ca_fait_reflechir = []

        while fait < N:
            x,y = x0 + kx, y0 + ky
            if self.test_inside(x,y) :
                if completion[x,y] != 1 and self.test_completable((x,y),grille):
                    completion[x,y] = 1
                elif completion[x,y] == 0 and grille[x,y] != 10:
                    completion[x,y] = -1

                if completion[x,y] == -1:
                    ca_fait_reflechir.append((x,y))


                if ANIMATE:
                    if completion[x,y] == 1:
                        self.demineur.animation((x,y),"vert")
                    elif completion[x,y] == -1:
                        self.demineur.animation((x,y),"bleu")
                    elif completion[x,y] == 0:
                        self.demineur.animation((x,y),"rouge")

                fait += 1
            n,kx,ky,etat = self.tourne_en_rond((n,kx,ky,etat))
            self.demineur.coloriage()

        return completion,ca_fait_reflechir

    def liste_voisins(self,slots):

        if not isinstance(slots, list):
            slots = [slots]

        voisin = []
        for slot in slots:
            x,y = slot
            for slot_ in [(x+i,y+j) for i in range(-1,2) for j in range(-1,2) if self.test_inside(x+i,y+j)]:
                if slot_ not in voisin and slot_ not in slots:
                    voisin.append(slot_)

        return voisin
    
    def all_k_among_n(self,k,n):

        if k > n:
            return []

        elif n == 1:
            return [[k]]
        
        elif k == 0:
            return [[0]*n]

        else :
            return [[1]+i for i in self.all_k_among_n(k-1,n-1)] + [[0]+i for i in self.all_k_among_n(k,n-1)]
        

    def find_good_completable(self,grille,completable_slots):

        
        slots = []

        for slot in completable_slots:
            n = self.compteur_rien(slot,grille)
            drapeau = self.compteur_drapeau(slot,grille)
            k = grille[slot] - drapeau
            if k > n or k < 0:
                print("on a détecté une incohérence")
                print(K)
            
            elif k == 0:
                return [(slot,1)],-1 

            comb = combinaison(n,k)
            if comb == 1:
                return [(slot,1)],1
            
            else:
                slots.append((slot,comb))
        
        return slots,0


    def check_completable(self,slots,grille,k = True,reel = True):

        if not isinstance(slots, list):
            slots = [slots]

        edited = []
        double_check = []

        for slot in slots:

            if self.test_inside(*slot) :
                if grille[slot] in range(1,9) and grille[slot] == self.compteur_drapeau(slot,grille) and self.compteur_rien(slot,grille) > 0:
                    grille,edited = self.demineur.ouverture_droite(slot,grille,reel,edited)
                    self.demineur.update_display(grille)
                elif k and grille[slot] == 10 and slot not in double_check:
                    double_check.append(slot)
        if k:    
            grille,_ = self.check_completable(double_check,grille,k = False)
        
        return grille,edited

    def action(self,slots, grille, completion, reel = True):
        
        slots_incomplet = [slot for slot in slots if completion[slot] == -1]

        slots_utiles = []
        for slot in slots_incomplet:
            
            rien = self.compteur_rien(slot,grille)
            q = grille[slot] - (rien + self.compteur_drapeau(slot,grille))
            
            if q > 0: #Il n'y a passe assez de cases vides pour placer les drapeaux : l'hypothèse est fausse
                return slots_utiles,grille,False
            elif not q < 0 and rien > 0:
                slots_utiles.append(slot)

        drapeau = [slot for slot in self.liste_voisins(slots_utiles) if grille[slot] == 10]

        return *self.drapeau_potentiel(drapeau,grille,reel = reel),True

        
         







    def drapeau_potentiel(self,drapeau,grille,reel = True):

        if not isinstance(drapeau, list):
            drapeau = [drapeau]

        check = []
        drapeau_pose = []
        for d in drapeau:
            grille,true_flags = self.demineur.place_drapeau(d,grille,reel = reel)
            if true_flags :
                drapeau_pose.append(d)

        self.demineur.update_display(grille)

        for d in drapeau_pose:
            for v in self.liste_voisins(d):
                if v not in check:
                    check.append(v)


        grille,edited = self.check_completable(check,grille,reel = reel)

        if not reel:
            edited += drapeau_pose
        
        return grille,edited

    def best_choice(self,grille_proba,completion,P):

        p_min = np.min(grille_proba)
        slot = np.argwhere(grille_proba == p_min)[-1]
        if p_min != P:
            return slot
        
        diagonale = [(x,y) for x in [-1,1] for y in [-1,1]]
        frontalier = [(x,y) for x in range(-1,2) for y in range(-1,2) if x*y == 0 and x+y != 0]
        slot_incomplet = [(x,y) for x in range(self.X) for y in range(self.Y) if completion[x,y] == -1]
        slot_debat = [slot for slot in self.liste_voisins(slot_incomplet) if completion[slot] == 0]
        slot_choix = [slot for slot in self.liste_voisins(slot_debat) if grille_proba[slot] == P]


        for slot0 in slot_choix:
            bool_frontalier = False
            for slot1 in slot_debat:
                for front in frontalier:
                    if slot0[0] + front[0] == slot1[0] and slot0[1] + front[1] == slot1[1]:
                        bool_frontalier = True
                        break
                        
            if not bool_frontalier:
                for slot1 in slot_debat:
                    for diag in diagonale:
                        if slot0[0] + diag[0] == slot1[0] and slot0[1] + diag[1] == slot1[1]:
                            return slot0
                                   
        return slot_choix[np.random.randint(len(slot_choix))]
                



    def compute_proba(self,grille,completion):

        ferme = (completion == 0)
        semi_complet = [(x,y) for x in range(self.X) for y in range(self.Y) if completion[x,y] == -1]

        P = (self.nb_bombe - np.sum(grille.grille == 11))/np.sum(ferme)


        
        
        slot_interessant = [slot for slot in self.liste_voisins(semi_complet) if completion[slot] == 0]

        grille_proba = np.ones((self.X,self.Y))*P
        grille_proba[~ferme] = 1

        for slot in slot_interessant:
            grille_proba[slot] = 0
        
        for slot in slot_interessant:
            p = len(grille.children)
            for children in grille.children:
                q = len(children)
                for child in children:
                    if child[slot] == -11:
                        grille_proba[slot] += 1/(p*q)
        
        return grille_proba,P
                    

        


    def realisation(self,grille,completion):

        N = self.X*self.Y
        x0,y0 = self.coups[-1]
        fait = 0
        
        n,kx,ky,etat = 0,0,0,[0,0]

        while fait < N:
            x,y = x0 + kx, y0 + ky
            if self.test_inside(x,y) :
                if completion[x,y] != 1 and grille[x,y] < 0:
                    if grille[x,y] == -10:
                        grille = self.demineur.ouverture_gauche((x,y),grille)[0]

                        if ANIMATE:
                            self.demineur.animation((x,y),"vert")
                    else :
                        grille[x,y] = 11
                        if ANIMATE:
                            self.demineur.animation((x,y),"bleu")
                elif ANIMATE:
                    self.demineur.animation((x,y),"rouge")
                
                if ANIMATE:
                    self.demineur.coloriage()

                fait += 1
            n,kx,ky,etat = self.tourne_en_rond((n,kx,ky,etat))

        return grille



    def f(self,G, grille, *args):
        retour = G(*args)
        self.demineur.update_display(grille)
        return retour

    def play(self,grille = None,signe = 1,slot = None):
        global K
        global ANIMATE
        K += 1
        if False and K == 77:
            ANIMATE = True



        if grille is None:
            grille = self.grille_affichage.copy()
            

        completion = self.grille_completion.copy()


        self.demineur.update_display(grille)


        self.demineur.test_leave()
        self.demineur.update_display(grille)
        completion,ou_travailler = self.f(self.update_completion,grille,grille,completion)



        slots,deterministe = self.f(self.find_good_completable,grille,grille,ou_travailler)

        if len(slots) == 0:
            self.demineur.grille_affichage_valeur = grille.grille.astype(np.uint8)
            self.demineur.update_display(grille,force = True)
            self.demineur.gagner()
            t.sleep(1)
            return True


        if deterministe == 1:
            
            where_check = self.liste_voisins(slots[0][0])
            grille,_ = self.drapeau_potentiel(where_check,grille)
            self.demineur.update_display(grille)

            if signe == 1:
                self.grille_completion = completion
                self.grille_affichage = grille
                return self.play()
            else:
                return grille
        
        elif deterministe == -1:

            grille = self.check_completable(slots[0][0],grille)[0]
            self.demineur.update_display(grille)

            if signe == 1:
                self.grille_completion = completion
                self.grille_affichage = grille
                return self.play()
            else:
                return grille,True


            
        else:

            if signe == -1:
                print("Une seule hypothèse ne suffit pas, je pense que faut quand meme appliquer le process a certains endroit en transformant des 10 en -10")
                return None,False
            
            slots = sorted(slots,key=lambda x: -x[1])



            while len(slots) > 0:

                slot,comb = slots.pop(-1)
                rien = self.liste_rien(slot,grille)
                n = len(rien)
                k = grille[slot] - self.compteur_drapeau(slot,grille)
                all_comb = self.all_k_among_n(k,n)

                
                children = []

                for comb_ in all_comb:
                    where_check = []
                    child = grille.copy()
                    child.parent = grille
                    for ni,slot_rien in enumerate(rien):
                        if comb_[ni] == 1:
                            child,edited = self.drapeau_potentiel(slot_rien,child,False)
                            for slot_ in edited:
                                self.demineur.maj.append(slot_)
                                if slot_ not in where_check:
                                    where_check.append(slot_)
                        else:
                            child[slot_rien] = -10
                            self.demineur.maj.append(slot_rien)
                            if slot_rien not in where_check:
                                where_check.append(slot_rien)
                    self.demineur.update_display(child)
                    hyp_valide = True

                    while len(where_check) > 0 and hyp_valide:
                        
                        where_check = self.liste_voisins(where_check)
                        child,where_check,hyp_valide = self.f(self.action,child,where_check,child,completion,False)
                        self.demineur.update_display(child)
                        K += 1
                    
                    #Il faut faire que si on sort avec ok == False, on doit inverser l'hypothèse d'une certaine manière
                    if hyp_valide:
                        children.append(child)
                        self.demineur.update_display(child)
                    else :
                        comb -= 1

                    self.demineur.update_display(grille)

                if comb == 1:
                    
                    deterministe = 1
                    grille = self.f(self.realisation,grille,children[-1],completion)
                    slots = [] #break
                    #self.demineur.update_display(grille,force = True)

                else :
                    grille.children.append(children)
                

            if deterministe == 0:
                grille_proba,P = self.compute_proba(grille,completion) #Il faut tenir compte du nombre de bombe, y'a des choses a revoir
                best_slot = self.best_choice(grille_proba,completion,P)
                self.demineur.update_display(grille, force = True)

                self.demineur.grille_affichage_valeur = grille
                grille = self.demineur.ouverture_gauche(best_slot,grille)[0]


                if self.demineur.perdu:
                    t.sleep(1)
                    return False

            
            self.grille_affichage = grille
            return self.play(grille)



                
                



                                












        


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

FORCE = False

if False:
    a = np.array([[2, 3, 8],
       [3, 2, 3],
       [5, 0, 6]])
    b = np.array([[ 6,  1, 10],
       [ 1,  6,  1],
       [ 8,  4,  9]])
    
    A = Grille(3,3,a)
    B = Grille(3,3,b)
    print((A + B).grille)
    exit()

if __name__ == "__main__":
    if FORCE:
        X,Y = GRILLE.shape
        nb_bombe = int(np.sum(GRILLE))

    else:
        X,Y = 26,19
        nb_bombe = 99
        GRILLE = np.array([])

    nb_gagnes = 0
    nb_perdus = 0
    for _ in range(100):
        bot = Bot(X,Y,nb_bombe,GRILLE,FORCE)
        gagner = bot.play()
        if gagner:
            nb_gagnes += 1
        else:
            nb_perdus += 1
    print("Gagné : ",nb_gagnes)
    print("Perdu : ",nb_perdus)
    print("Pourcentage de réussite : ",nb_gagnes/(nb_gagnes+nb_perdus)*100)
