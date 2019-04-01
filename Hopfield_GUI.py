import tkinter as tk
import numpy as np
from functools import partial


class GridWindow:
    """Class based on Tkinter Canvas for creating and manipulating grids.

    The __init__ method has following arguments and propreties:
    Arguments:
        parent (<tkinter.Tk object .>): instance of tkinter object.
    
    Attributes:
        myContainer1 (<tkinter.Frame object .!frame>): tk.Frame object.
        cellwidth (int): width of one grid unit.
        cellheight (int): height of one grid unit.
        rect (dict): dictionary container to acces units.

    """
    def __init__(self, parent):
        self.myParent = parent
        self.myContainer1 = tk.Frame(parent,width = 800)
        self.cellwidth = 25
        self.cellheight = 25
        self.rect = {}

    def draw_grid(self, rows, columns):
        self.myCanvas = tk.Canvas(self.myContainer1)
        self.myCanvas.configure(width=self.cellheight*rows,
                                height=self.cellwidth*columns)
        self.myCanvas.pack(side=tk.RIGHT)

        for column in range(rows):
            for row in range(columns):
                x1 = column * self.cellwidth
                y1 = row * self.cellheight
                x2 = x1 + self.cellwidth
                y2 = y1 + self.cellheight
                self.rect[row, column] = self.myCanvas.create_rectangle(x1, y1, x2, y2, fill="white")
        return self.rect        
                
    def fill_grid(self,vector):
        degree =  int(np.sqrt(vector.shape[1]))
        k = 0
        for i in range(degree):
            for j in range(degree):
                k+=1
                if vector[0,k-1] == 1:
                    self.myCanvas.itemconfig(self.rect[i,j],fill='black')
                else:
                    self.myCanvas.itemconfig(self.rect[i,j],fill='white')
                    
        
        
           


class Hopfield():
    """Class to learn and evaluate Hopfield neural network.

    The __init__ method has following arguments and propreties:
        
    Attributes:
        x (numpy.ndarray): input vector, which is used to evaluate weight matrix.
        deg (int): degree of input square matrix.
        corrupted (numpy.ndarray): vector of distorted matrix which is to be recognized.
        n (int): number of rows of x, recognizable shapes.
        W (numpy.ndarray): weight matrix of Hopfield network.
        history (numpy.ndarray): history of iterations over corrupted vector used for graphical purposes.

    """
    
    def __init__(self):
        self.x = np.zeros(10)
        self.deg = 0 
        self.corrupted = np.zeros(self.deg).reshape((1,self.deg))
        self.n = 0 
        self.W = np.zeros((self.deg**2,self.deg**2))
        self.history = np.zeros(self.deg**2).reshape((1,self.deg**2))
        
    def addVector(self,vector):
        if self.x.any() == 0:
            self.x =  vector
            self.deg = int(np.sqrt(self.x.shape[1]))
            self.n = self.x.shape[0]
            self.W = np.zeros((self.deg**2,self.deg**2))
            self.history = np.zeros(self.deg**2).reshape((1,self.deg**2))
        else:
            self.x = np.vstack((self.x,vector))
            self.n = self.x.shape[0]
        return self.x    
        
        
    def learn(self):
        for i in range(self.deg*self.deg):
            for j in range(self.deg*self.deg):
                weight = 0
                if not(i==j):
                    for n in range(self.n):
                        weight = self.x[n,i]*self.x[n,j] + weight
                self.W[i,j] = weight
        return self.W 
    
    def recognize(self,L_iteraciVar,RightGrid,root):
        iteration = 0
        flag = True
        iterationOfLastChange = 0
                
        while flag:
            iteration +=1
            L_iteraciVar.set('pocet iteraci: '+ str(iteration))
            suma = 0
            i = np.random.randint((self.deg**2)-1)
            
            for j in range(self.deg**2):
                suma  = self.W[i,j]*self.corrupted[0,j] + suma
                
            out = 0
            changed = 0
            
            if suma != 0:
                if suma < 0:
                    out = -1
                elif suma > 0:
                    out = 1
                if out != self.corrupted[0,i]:
                    changed = 1
                    self.corrupted[0,i] = out
                    RightGrid.fill_grid(self.corrupted)
                    self.history = np.vstack((self.history,self.corrupted))
                                        
            if changed == 1:
                iterationOfLastChange = iteration
            if (iteration-iterationOfLastChange)>1000:
                flag = False
            
        return self.history        
                
            

          
                    
                    
    


class Transform:
    """Class for transformation and manipulation with input .txt.

    The __init__ method has following arguments and propreties:
        
    Attributes:
        filename (str): name of file .txt
        Binary (numpy.ndarray): initialized as (int 0) stores Binary represaentation of .txt grid.
        Corrupted (numpy.ndarray): stores distorted vector.

    """
    def __init__(self,filename):
        self.filename = filename+'.txt'
        self.Binary = 0
        self.Corrupted = np.zeros((1,10))

    def FileToBinary(self):
        
        with open(self.filename) as fp:
            line = fp.readline()
            cnt = 1
            vect = []
            
            while line:
                vect.append(list(line.strip()))
                line = fp.readline()
                cnt += 1
                
            n = len(vect)    
            Binary = list(range(n))
            
            for i in range(n):
                Binary[i] = [ 1 if x=='x' else -1 for x in vect[i]]
                
            self.Binary = np.array(Binary)
            
            return self.Binary
        
    def Corrupt(self, rate):
        deg = self.Binary.shape[0]
        C = np.random.random_sample((deg,deg)) < rate
        self.Corrupted = np.where(np.logical_not(C),self.Binary,np.negative(self.Binary))
        
        return self.Corrupted
    
    def Flatten(self):
        rank=int(self.Binary.shape[0])
        self.Binary=self.Binary.flatten().reshape(1,rank**2)
        
        if np.all(self.Corrupted) != 0:
            self.Corrupted=self.Corrupted.flatten().reshape(1,rank**2)
            return self.Binary, self.Corrupted
        else:
            return self.Binary
            
        
        
        




def runApp(rows, columns):
    #konstrukce okna
    root = tk.Tk()
    Hop = Hopfield()
    root.geometry("700x500")
    root.resizable(0,0)
    #citac iteraci
    L_iteraciVar = tk.StringVar()
    L_iteraci = tk.Label(root, textvariable=L_iteraciVar, fg="black").grid(column=1,sticky=tk.N)
    L_iteraciVar.set('iteraci: 0')
    #zpravy
    L_zpravyVar = tk.StringVar()
    L_zpravy = tk.Label(root, textvariable=L_zpravyVar, fg="black").grid(row=5,column=1,sticky=tk.S)
    
    #Leve okno
    LeftGrid = GridWindow(root)
    LeftGrid.myContainer1.grid(row=0, sticky=tk.W)
    LeftGrid.draw_grid(rows, columns)
    #Prave okno
    RightGrid = GridWindow(root)
    RightGrid.myContainer1.grid(row=0, column =2,sticky=tk.E)
    RightGrid.draw_grid(rows, columns)
    #ENTRY soubor
    L_vstup = tk.Label(root, text="filename: ", fg="black").grid(row=1,column=0,sticky=tk.W)
    vstup = tk.Entry(root)
    vstup.grid(row=1,column=0,sticky=tk.N)
    #tlacitko LOAD
    B_load = tk.Button(root, text="Load", command=partial(load,Hop,LeftGrid,vstup,L_zpravyVar))
    B_load.grid(row=2)
    #Spinbox na volbu poruchy
    L_porucha = tk.Label(root, text="corruption:", fg="black").grid(row=1,column=2,sticky=tk.W)
    myvar = tk.StringVar()
    myvar.set('20')
    S_porucha = tk.Spinbox(root,textvariable=myvar, from_=0, to=100)
    S_porucha.grid(row=1, column =2,sticky=tk.E)
    myvar.trace('w',partial(corrupt,Hop,RightGrid,vstup,L_zpravyVar,myvar))
    # RUN tlacitko
    B_Run = tk.Button(root, text="Run", command=partial(run,RightGrid,vstup,L_zpravyVar,Hop,L_iteraciVar,root))
    B_Run.grid(row=3)
    
            
        
    
    
    
    
    
    
    root.mainloop()
 
def run(RightGrid,vstup,L_zpravyVar,Hop,L_iteraciVar,root):
    soubor = vstup.get()
    if len(soubor) == 0:
        L_zpravyVar.set('Nebyl zadan nazev souboru.')
    else:
        Hop.learn()
        t = Transform(soubor)
        t.FileToBinary()
        t=t.Flatten()
        rec=Hop.recognize(L_iteraciVar,RightGrid,root)
        #kod na vykresleni do Praveho Gridu
        def callback():
            nonlocal root, rec, RightGrid
            if rec.shape[0] == 0:
                pass
            else:
                root.after(200,RightGrid.fill_grid(rec[0].reshape((1,rec.shape[1]))))
                rec = np.delete(rec,0,0)
                root.after(200, callback)
        root.after(200, callback)  
    
    

def load(Hop,LeftGrid,vstup,L_zpravyVar):
        soubor = vstup.get()
        if len(soubor) == 0:
            L_zpravyVar.set('Nebyl zadan nazev souboru.')
            
        else:
            t = Transform(soubor)
            t.FileToBinary()
            t=t.Flatten()
            Hop.addVector(t)
            #kod na vykresleni do Leveho Gridu
            LeftGrid.fill_grid(t)

def corrupt(Hop,RightGrid,vstup,L_zpravyVar,myvar,*args):
    soubor = vstup.get()
    
    if len(soubor) == 0:
        L_zpravyVar.set('Nebyl zadan nazev souboru.')
    else:
        t = Transform(soubor)
        t.FileToBinary()
        t.Corrupt(float(myvar.get())/100)
        t.Flatten()
        t = t.Corrupted
        Hop.corrupted = t
        #kod na vykresleni do Praveho Gridu
        RightGrid.fill_grid(t)

if __name__ == '__main__':
    runApp(10, 10)
    
    