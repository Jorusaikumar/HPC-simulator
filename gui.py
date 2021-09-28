# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:16:35 2021

@author: saikumar
"""

import Tkinter as tk
import tkFont
import tkMessageBox as mb
import tkFileDialog as fd
import os
import re
import shutil

path=os.path.dirname(os.path.abspath(__file__))

class DagGui(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("DAG Generation")
        self.geometry("1200x700")
        self.resizable(True,True)        
        container = tk.Frame(self)
        container.pack(side='top',fill='both',expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)        
        self.frames = {}
        for F in (KernelPage,EdgePage):
            page_name = F.__name__
            frame = F(parent=container,controller=self)
            self.frames[page_name]=frame
            frame.grid(row=0, column=0, sticky='nsew')        
        self.show_frame("KernelPage")
        
    def build_kernelspage(self):
        self.frames["EdgePage"].add_nodes(self.frames["KernelPage"].get_selected_kernels())
        self.show_frame("EdgePage")
        
    def backto_kernelspage(self):
        self.frames["EdgePage"].clear_canvas()
        self.show_frame("KernelPage")
        
    def show_frame(self,page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        
    def quit_app(self):
        self.destroy()

    def submit_json(self):
        self.frames["EdgePage"].create_json()
        self.destroy()


        
class KernelPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        label1 = tk.Label(self,text="Select kernels from the list below    (OR)",bg='orange',fg='black',width=45)
        label1.config(font=("Courier", 15))
        label1.place(x=200,y=10)
        label2 = tk.Label(self,text="1)Click to select the kernel\t\t\n2)Double click to preview the kernel")
        label2.place(x=350,y=50)
        frame1=tk.Frame(self)
        frame1.place(x=200,y=90)
        self.list_kernels=os.listdir(path+'/database/kernels')
        self.list_box1 = tk.Listbox(frame1,selectmode='multiple',width=100,height=25)
        self.list_box1.pack(side='left',fill='y')
        scrollbar1 = tk.Scrollbar(frame1,orient='vertical')
        scrollbar1.pack(side='right',fill='y')
        for item in range(len(self.list_kernels)):
            self.list_box1.insert('end',self.list_kernels[item])
            self.list_box1.itemconfig(item,bg='pink')
        self.list_box1.config(yscrollcommand = scrollbar1.set)
        scrollbar1.config(command = self.list_box1.yview)
        self.list_box1.bind('<Double-1>',self.preview)
        
        #Button for submitting list of selected kernels
        button1=tk.Button(self,text="Next",width=10,fg='black',bg='skyblue',command=controller.build_kernelspage)
        button1.place(x=450,y=500)
        button2=tk.Button(self,text="Close",width=10,fg='black',bg='skyblue',command=controller.quit_app)
        button2.place(x=550,y=500)
        button3 = tk.Button(self,text="Choose from folder",fg='black',bg='orange',command=self.filemanager)
        button3.place(x=750,y=10,width=150,height=30)
        self.filenames =[]
        
    def filemanager(self):
        directory = fd.askdirectory(initialdir="/database/kernels",title="Select files")
        self.list_box1.delete(0,'end')
        files = os.listdir(directory)
        i=0
        for item in files:
            self.list_box1.insert('end',item)
            self.list_box1.itemconfig(i,bg='pink')
            i+=1
            
        """for item in files:
            L=item.split("/")
            self.filenames.append(L[-1])"""
        
                
    def preview(self,name):
        cs = self.list_box1.get('active')
        kernel_file = open(path+'/database/kernels/'+cs)
        code = kernel_file.read()
        kernel_file.close()
        code = "The following kernel is selected.Click again to deselect\n--------------------------------------------------------------------\n\n\n"+code
        mb.showinfo(cs,code)
        
    def get_selected_kernels(self):
        indices = self.list_box1.curselection()
        self.filenames = []
        for i in indices:
            self.filenames.append(self.list_kernels[i])
        return self.filenames
        #return self.list_box1.curselection()
    
    def get_all_kernels(self):
        return self.list_kernels





class EdgePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.parent = parent
        self.controller = controller
        label1 = tk.Label(self,text="Choose dependencies",bg='orange',fg='black',width=50,height=1)
        label1.config(font=("Courier", 20))
        label1.place(x=80,y=10)        
        button1=tk.Button(self,text="Ok",width=10,fg='black',bg='skyblue',command=controller.submit_json)
        button1.place(x=400,y=670)
        button2=tk.Button(self,text="Back",width=10,fg='black',bg='skyblue',command=controller.backto_kernelspage)
        button2.place(x=500,y=670)
        button3=tk.Button(self,text="Close",width=10,fg='black',bg='skyblue',command=controller.quit_app)
        button3.place(x=600,y=670)
        button4=tk.Button(self,width=5,text="edge",fg='black',bg='white',command = lambda: self.set_tool('src'))
        button4.place(x=1060,y=50)
        button5=tk.Button(self,width=5,text="Esc",fg='black',bg='white', command = lambda: self.set_tool('inactive'))
        button5.place(x=1100,y=50)
        button6=tk.Button(self,width=5,text='Move',fg='black',bg='white', command = lambda: self.set_tool('move'))
        button6.place(x=1140,y=50)        
        self.label2 = tk.Label(self,fg='black',bg='yellow')
        self.label2.place(x=1060,y=100,width=140)        
        self.canvas = tk.Canvas(self,bg='white',width=1000,height=600)
        self.canvas.place(x=50,y=50)
        self.canvas.bind("<ButtonPress-1>",self.move_kernel)        
        self.buttons_data = {}        
        self.status = 'inactive'
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.k1 = None
        self.b1 = None
        self.k2 = None
        self.b2 = None
        self.moving_kernel = None
        self.dag = {}
        
    def add_nodes(self,selected_kernels):
        shutil.rmtree('kernel_info')
        os.makedirs(path+'/kernel_info')
        
        circle_rad = 15
        count=0
        label_text="NOTATIONS\n\n"
        for item in selected_kernels:
            kernel_name = item[:-3]
            label_text+="k"+str(count)+" : "+kernel_name+".cl\n"
            feat_file = open(path+'/database/features/'+kernel_name+'.feat')
            shutil.copy(path+'/database/info/'+kernel_name+'.json',path+'/kernel_info')
            feat=feat_file.readlines()
            feat_file.close()
            inputs=[]
            outputs=[]
            for line in feat:
                L=re.split(' -> |_|,',line)
                if(len(L)!=4):
                    break
                if(L[2]=='input'):
                    inputs.append((L[0],L[1]))
                elif(L[2]=='output'):
                    outputs.append((L[0],L[1]))
                elif(L[2]=='io'):
                    inputs.append((L[0],L[1]))
                    outputs.append((L[0],L[1]))
            row = int(count/5)
            column = int(count%5)
            cx = 100+column*200
            cy = 60+row*120
            button = tk.Button(self,text="k"+str(count),bg='green',command = lambda index=count,name=kernel_name: self.parameters(index,name))
            button.place(x = 50+cx-circle_rad, y = 50+cy-circle_rad, width = 2*circle_rad, height = 2*circle_rad)
            self.dag[count]=[]
            self.dag[count].append(str(count)+" "+kernel_name+".json")
            kernel_buts = {}
            kernel_buts['cx'] = cx
            kernel_buts['cy'] = cy
            kernel_buts['node'] = button
            kernel_buts['input'] = []
            kernel_buts['output'] = []
            kernel_buts['lines'] = []
            
            b_count=0
            n=len(inputs)
            for b in inputs:
                x0=cx+40*(b_count-int(n/2))-15
                y0=cy-45
                but = tk.Button(self,text="b"+b[0],bg='grey',fg='black',command=lambda x=x0+12,y=y0+7,k=count,bi=b[0]: self.select_dest(x,y,k,bi))
                but['font']=tkFont.Font(size=7)
                but.place(x=x0+50,y=y0+50,width=25,height=15)
                kernel_buts['input'].append(but)
                line = self.canvas.create_line(x0+13,y0+15,cx,cy-circle_rad,arrow=tk.LAST)
                kernel_buts['lines'].append(line)
                b_count+=1
            b_count=0
            n=len(outputs)
            for b in outputs:
                x0=cx+40*(b_count-int(n/2))-15
                y0=cy+30
                but = tk.Button(self,text="b"+b[0],bg='grey',fg='black',command=lambda x=x0+12,y=y0+7,k=count,bi=b[0]: self.select_src(x,y,k,bi))
                but['font']=tkFont.Font(size=7)
                but.place(x=x0+50,y=y0+50,width=25,height=15)
                kernel_buts['output'].append(but)
                line = self.canvas.create_line(cx,cy+circle_rad,x0+13,y0,arrow=tk.LAST)
                kernel_buts['lines'].append(line)
                b_count+=1
            self.buttons_data[count] = kernel_buts
            count+=1
        self.dag['dependencies']=[]
        self.label2.config(text=label_text)
        
        
    def clear_canvas(self):
        self.canvas.delete("all")
        for kernel,buts in self.buttons_data.items():
            buts['node'].destroy()
            for item in buts['input']:
                item.destroy()
            for item in buts['output']:
                item.destroy()
        
    def set_tool(self,tool):
        self.status = tool
                
    def select_src(self,x,y,k,bi):
        if self.status == 'src':
            self.x1 = x
            self.y1 = y
            self.k1 = k
            self.b1 = bi
            self.status = 'dest'
        elif self.status == 'inactive':
            mb.showerror("Error","No tool selected!!")
        elif self.status == 'move':
            mb.showerror("Error","Click on kernel node to move")
        else:
            if(mb.askyesno("Warning","Do you want to change the source?")):
                self.x1 = x
                self.y1 = y
                self.k1 = k
                self.b1 = bi
            else:
                pass
                    
    def select_dest(self,x,y,k,bi):
        if self.status == 'dest':
            self.x2 = x
            self.y2 = y
            self.k2 = k
            self.b2 = bi
            self.canvas.create_line(self.x1,self.y1,self.x2,self.y2,arrow = 'last',fill='red',width=2,arrowshape=(25,30,5))
            self.canvas.after(1)
            self.canvas.update()
            print(self.k1,self.b1,self.k2,self.b2)
            self.dag['dependencies'].append(str(self.k1)+" "+str(self.b1)+"-"+str(self.k2)+" "+str(self.b2))
            self.status = 'src'
            self.k1 = None
            self.b1 = None
            self.k2 = None
            self.b2 = None
        elif self.status == 'inactive':
            mb.showerror("Error","No tool selected!!")
        elif self.status == 'move':
            mb.showerror("Error","Click on kernel node to move")
        else:
            mb.showerror("Error","Please, select the source first.")
            
    def parameters(self,index,name):
        if(self.status == 'move'):
            self.moving_kernel = index
        else:
            v1=""
            v2=""
            if(len(self.dag[index])!=1):
                v1,v2 = self.dag[index][1],self.dag[index][2]
            
            win = tk.Toplevel(self)
            win.title("Parameters")
            win.geometry("400x500")
            label0 = tk.Label(win,text="Enter parameters for kernel k"+str(index)+"("+name+".cl)",bg='purple',fg='black')
            label0.place(x=0,y=0,width=400,height=30)
            label1 = tk.Label(win,text="Dataset",bg='grey',fg='black')
            label1.place(x=20,y=400,width=100,height=20)
            entry1 = tk.Entry(win)
            entry1.place(x=130,y=400,width=100,height=20)
            entry1.insert(0,v1)
            label2 = tk.Label(win,text="Partition",bg='grey',fg='black')
            label2.place(x=20,y=430,width=100,height=20)
            entry2 = tk.Entry(win)
            entry2.place(x=130,y=430,width=100,height=20)
            entry2.insert(0,v2)
            but2 = tk.Button(win,text="Close",bg='skyblue',fg='black',command = win.destroy)
            but2.place(x=220,y=470)
            win.attributes('-topmost','true')
            
            frame1 = tk.Frame(win)
            frame1.place(x=20,y=50)
            scrollbar = tk.Scrollbar(frame1)
            text_edit = tk.Text(frame1,width=40,height=20,yscrollcommand=scrollbar.set)
            scrollbar.config(command=text_edit.yview)
            scrollbar.pack(side='right', fill='y')
            text_edit.pack(side='left')
            json_file = open(path+"/kernel_info/"+name+".json")
            content = json_file.read()
            json_file.close()
            text_edit.insert('end',content)
            but1 = tk.Button(win,text="Save",bg='skyblue',fg='black',command = lambda index=index,e1=entry1,e2=entry2,text=text_edit,name=name: self.save_parameters(index,name,e1,e2,text))
            but1.place(x=160,y=470)
            win.mainloop()
            
    def save_parameters(self,index,name,e1,e2,text):
        dataset = e1.get()
        partition = e2.get()
        if(len(self.dag[index])==1):
            self.dag[index].append(dataset)
            self.dag[index].append(partition)
        else:
            self.dag[index][1] = dataset
            self.dag[index][2] = partition
        content = text.get(1.0,'end')
        json_file = open(path+"/kernel_info/"+name+".json","w")
        json_file.write(content)
        json_file.close()
            
    def move_kernel(self,event=None):
        if self.moving_kernel is not None:
            newX = event.x
            newY = event.y
            oldX = self.buttons_data[self.moving_kernel]['cx']
            oldY = self.buttons_data[self.moving_kernel]['cy']
            self.buttons_data[self.moving_kernel]['node'].place_configure(x=newX+35,y=newY+35)
            inputs = self.buttons_data[self.moving_kernel]['input']
            outputs = self.buttons_data[self.moving_kernel]['output']
            lines = self.buttons_data[self.moving_kernel]['lines']
            b_count=0
            n=len(inputs)
            for b in inputs:
                x0=newX+40*(b_count-int(n/2))-15
                y0=newY-45
                b.place_configure(x=x0+50,y=y0+50)
                b.config(command=lambda x=x0+12,y=y0+7,k=self.moving_kernel,bi=b['text'][1:]: self.select_dest(x,y,k,bi))
                b_count+=1
            b_count=0
            n=len(outputs)
            for b in outputs:
                x0=newX+40*(b_count-int(n/2))-15
                y0=newY+30
                b.place_configure(x=x0+50,y=y0+50)
                b.config(command = lambda x=x0+12,y=y0+7,k=self.moving_kernel,bi=b['text'][1:]: self.select_src(x,y,k,bi))
                b_count+=1
            for item in lines:
                self.canvas.move(item,newX-oldX,newY-oldY)
            self.canvas.update()
            self.moving_kernel = None
                    
    def create_json(self):
        content = ""
        for index,item in self.dag.items():
            if index == 'dependencies':
                content += "---\n"
                for e in item:
                    content += e + "\n"
            else:
                content += item[0] + ' {"dataset":' + item[1] + ',"partition":' + item[2] + ',"i":' + 'None' + '}' +"\n"
        content += "---\n"
        dag_file = open(path+'/dag.json','w')
        dag_file.write(content)
        dag_file.close()

app = DagGui()
app.mainloop()