# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:41:29 2020

@author: Henry Lu
"""
import time
import numpy as np
import plotly.graph_objects as go

print("Henry Lu's FEM Module 已导入！")

# TODO:桁架/钢架体系FEM
class FEM:
# TODO:输入部分
    def __init__(self,geometric,unit,force,constraint=dict(NR=[],DX=[]),dim = 3):
        if dim == 3:
            self.dim = dim
            self.type = list(unit.keys())[0]
            self.err, self.largeNumber = 1e-10, 1e25
            self.geo = {}
            self.unit = {}
            self.con = {}
            self.force = {}
            self.K = None
            self.geo["DX"] = None
            self.geometric(geometric["X"])
            if self.type == "rod":
                self.rod(unit["rod"]["ME"],unit["rod"]["E"],unit["rod"]["A"])
            self.constraint(constraint["NR"],constraint["DX"])
            for i in force.keys():
                if i == "node":
                    self.nodeForce(force["node"])
            print("计算问题为",dim,"维问题...")
        elif dim == 2:
            raise NameError("暂未开发!")
        else: 
            raise NameError("维度有误！")
    
    def geometric(self,X=None):
        self.geo["X"] = np.array(X)
        if self.geo["X"].shape[0] == self.dim:
            print("一共",self.geo["X"].shape[1],"节点...")
        else:
            raise NameError("几何数据输入有误！")
        self.geo["L"] = (self.geo["X"].max(1)-self.geo["X"].min(1)).sum()/self.dim
                
    def rod(self,ME,E,A):
        self.unit["rod"] = {}
        self.unit["rod"]["ME"] = np.array(ME,dtype = 'int')-1
        n = self.unit["rod"]["ME"].shape[1]
        self.unit["rod"]["E"] = np.array(E)
        self.unit["rod"]["A"] = np.array(A)
        self.unit["rod"]["L"] = np.zeros(n)
        self.unit["rod"]["T"] = [[]]*n
        self.unit["rod"]["K"] = [[]]*n
        for i in range(n):
            node = self.unit["rod"]["ME"][:,i]
            DX = (self.geo["X"][:,node[1]] - self.geo["X"][:,node[0]])
            L = ( DX ** 2).sum() ** 0.5
            self.unit["rod"]["L"][i] = L
            self.unit["rod"]["T"][i] = np.array( [DX.tolist() + [0]*3, [0]*3 + DX.tolist()]) / L
            self.unit["rod"]["K"][i] = E[i] * A[i] / L * np.array([[1,-1],[-1,1]])
        if self.unit["rod"]["ME"].max() < self.geo["X"].shape[1] and self.unit["rod"]["ME"].min()>=0 \
            and n == self.unit["rod"]["E"].shape[0] \
            and n == self.unit["rod"]["A"].shape[0] :
            print("一共",n,"单元...")
        else:
            raise NameError("单元数据输入有误！")
        
    def constraint(self, NR, DX = None):
        self.con["NR"] = np.array(NR, dtype = "int") -1
        self.con["DX"] = np.zeros(self.dim * self.geo["X"].shape[1]) if DX == None else np.array(DX)
        print("一共",self.con["NR"].shape[0],"约束...")
        if self.con["NR"].max() >= self.dim * self.geo["X"].shape[1] or self.con["NR"].min()<0:
            raise NameError("节点约束数据输入有误！")
    
    def nodeForce(self, P):
        if len(P) == self.dim * self.geo["X"].shape[1]:
            self.force["node"] = np.array(P)
            print("一共",(self.force["node"]!=0).sum(),"节点外力...")
        else:
            raise NameError("节点力数据输入有误！")
        self.force["max"] = np.abs(self.force["node"]).max()
        
# TODO:生成输入报告
    def inputText(self,file=None):
        if self.type == "rod":
            print("几何输入数据:",file=file)
            for i in range(self.geo["X"].shape[1]):
                print("  节点 {:3} : X = {:4e} | Y = {:4e} | Z = {:4e} "\
                      .format( i,self.geo["X"][0,i],self.geo["X"][1,i],self.geo["X"][2,i]),
                      file=file)
            print("单元输入数据:",file=file)
            t = "rod"
            print("  杆单元数据:",file=file)
            for i in range(self.unit[t]["ME"].shape[1]):
                print("    杆单元 {:3} : 节点 = ({:2},{:2}) | E = {:4e} | A = {:4e} | L = {:4e}"\
                      .format(i,self.unit[t]["ME"][0,i],self.unit[t]["ME"][1,i], 
                              self.unit[t]["E"][i] , self.unit[t]["A"][i], self.unit[t]["L"][i]),
                      file=file)
            print("约束输入数据:",file=file)
            n = ["X","Y","Z"]
            for i in range(self.con["NR"].shape[0]):
                print("  约束 {:3} : 节点 {:2} | {} 方向 | 位移: {:4e}"\
                      .format( i, self.con["NR"][i] // 3, n[self.con["NR"][i] % 3], self.con["DX"][i]),
                      file=file)
            print("节点外力输入数据:",file=file)
            Fn = 0
            for i in range(self.force["node"].shape[0]):
                if self.force["node"][i] != 0:
                    print("  节点外力 {:3} : 节点 {:2} | {} 方向 | 大小: {:4e}".\
                          format( Fn , i // 3, n[i % 3], self.force["node"][i]),
                          file=file)
                    Fn += 1
        
    def inputFigure(self,nodeSize=1,lineSize=1,forceDisplaySize=1,
                    constraintDisplaySize=1,opa=dict(node = 0.3, rod = 0.6, constraint = 1, force = 0.6)):
        figs = []
        if not isinstance(opa,type({})):
            opa = dict(node = opa, rod = opa, constraint = opa, force = opa)
        # Add nodes
        figs += self.nodeFigure(nodeSize,opa["node"])
        # Add rods
        if self.type == "rod":
            figs += self.rodFigure(lineSize,opa["rod"])
            # Add rod constraint
            figs += self.rodConstraintFigure(constraintDisplaySize,opa["constraint"])
            # Add rod node force
            figs += self.rodNodeForceFigure(forceDisplaySize, opa["force"])
        return figs
        
    def inputReport(self,text=True,figure=True,file=None,nodeSize=1,lineSize=1,
                    forceDisplaySize=1,constraintDisplaySize=1,opacity=1):
        if text:
            self.inputText()
        if file != None:
            with open(file,'w+') as f:
                self.inputText(file=f)
            print("数据已保存至",file,"!")
        if figure:
            print("数据可视化:")
            fig = go.Figure(data=self.inputFigure(nodeSize,lineSize,
                                                  forceDisplaySize,constraintDisplaySize,opacity))
            # fig.update_layout(width = 1000, height = 800)
            fig.show()
    
# TODO:计算部分
    def calK(self,showResult=False):
        if self.type == "rod":
            K = diagMat(self.geo["X"].size)
            for i in range(self.unit["rod"]["ME"].shape[1]):
                T_ = self.unit["rod"]["T"][i]
                K_e = np.dot(T_.T,(self.unit["rod"]["K"][i]).dot(T_))
                n = self.unit["rod"]["ME"][:,i]
                I = [n[0]*3 + i_ for i_ in range(3)] + [n[1]*3 + i_ for i_ in range(3)]
                MK = np.abs(K_e).max()
                for ii in range(6):
                    for jj in range(ii+1):
                        if np.abs(K_e[ii,jj] / MK) > self.err:
                            K.change(I[ii],I[jj],K.value(I[ii],I[jj])+K_e[ii,jj])
        self.K = K
        if showResult:
            self.K.show()
        print("总体刚度矩阵K计算完毕!")
        return K
    
    def setLargeDiag(self):
        K = self.K
        P = self.force["node"]
        NR = self.con["NR"]
        Ks = self.largeNumber
        for i in range(NR.shape[0]):
            n = NR[i]
            P[n] = Ks * self.con["DX"][i]
            K.change(n,n,Ks)
        print("主角元置大数完成!")
        return K,P
    
    def calDX(self):
        K,P = self.setLargeDiag()
        DX = cholesky(K,P)
        # DX = mySolve(K, P)
        absDX = np.abs(DX)
        DX[absDX / absDX.max() < self.err] = 0
        self.geo["DX"] = DX
        print("节点位移计算完毕!")
        
    def calLocalDX(self):
        if self.type == "rod":
            self.unit["rod"]["LocalDX"] = []
            for i in range(self.unit["rod"]["ME"].shape[1]):
                n = self.unit["rod"]["ME"][:,i]
                I = [n[0]*3 + i_ for i_ in range(3)] + [n[1]*3 + i_ for i_ in range(3)]
                DX = self.geo["DX"][I]
                LocalDX = np.dot(self.unit["rod"]["T"][i], DX)
                self.unit["rod"]["LocalDX"].append(LocalDX)
        print("局部节点位移计算完毕!")
    
    def calUnitForce(self):
        self.unit[self.type]["force"] = []
        for i in range(self.unit[self.type]["ME"].shape[1]):
            K = self.unit[self.type]["K"][i]
            LocalDX = self.unit[self.type]["LocalDX"][i]
            force = np.dot(K, LocalDX)
            self.unit[self.type]["force"].append(force)
        print("单元内力计算完毕!")
        
    def calUnitNodeForce(self):
        self.unit[self.type]["nodeForce"] = []
        for i in range(self.unit[self.type]["ME"].shape[1]):
            T = self.unit[self.type]["T"][i]
            force = self.unit[self.type]["force"][i]
            nodeForce = np.dot(T.T, force)
            self.unit[self.type]["nodeForce"].append(nodeForce)
        print("单元节点力计算完毕!")
        
    def calConstraintForce(self):
        if self.type == "rod":
            self.force["constraint"] = np.zeros(self.geo["X"].size)
            for i in range(self.unit["rod"]["ME"].shape[1]):
                n = self.unit["rod"]["ME"][:,i]
                I = [n[0]*3 + i_ for i_ in range(3)] + [n[1]*3 + i_ for i_ in range(3)]
                self.force["constraint"][I] += self.unit["rod"]["nodeForce"][i]
        self.force["constraint"] = self.force["constraint"] - self.force["node"]
        self.force["max"] = max(self.force["max"], np.abs(self.force["constraint"]).max())
        self.force["constraint"][np.abs(self.force["constraint"]) / self.force["max"] < self.err] = 0
        print("约束力计算完毕!")
    
    def solve(self,showK = False, showResult=False):
        start = time.time()
        self.calK(showK)
        self.calDX()
        self.calLocalDX()
        self.calUnitForce()
        self.calUnitNodeForce()
        self.calConstraintForce()
        end = time.time()
        print("用时", round(end-start,4),"s。")
        if showResult:
            self.resultText()
    
# TODO:生成结果报告
    def resultText(self,file=None):
        n = ["X","Y","Z"]
        if self.type == "rod":
            print("节点位移结果:",file=file)
            for i in range(self.geo["DX"].shape[0]):
                print("  节点位移 {:3} : 节点 {:2} | {} 方向 | 大小: {:4e}"\
                      .format( i , i // 3, n[i % 3], self.geo["DX"][i]),file=file)
            print("单元内力结果:",file=file)
            t = "rod"
            print("  杆单元内力大小:",file=file)
            for i in range(self.unit[t]["ME"].shape[1]):
                print("    杆单元 {:3} : 节点{:2}内力 = {:4e} | 节点{:2}内力 = {:4e}"\
                      .format(i,self.unit[t]["ME"][0,i],self.unit[t]["force"][i][0],
                              self.unit[t]["ME"][1,i], self.unit[t]["force"][i][1]),
                      file=file)
            print("  杆单元节点力:",file=file)
            for i in range(self.unit[t]["ME"].shape[1]):
                F = self.unit[t]["nodeForce"][i]
                print(("    杆单元 {:3} : 节点{:2}内力 X={:4e} Y={:4e} Z={:4e} "
                      "| 节点{:2}内力 = X={:4e} Y={:4e} Z={:4e}")\
                      .format(i,self.unit[t]["ME"][0,i],F[0],F[1],F[2],
                              self.unit[t]["ME"][1,i],F[3],F[4],F[5]),
                      file=file)
            print("约束反力结果:",file=file)
            Fn = 0
            for i in self.con["NR"]:
                if self.force["constraint"][i] != 0:
                    print("  约束反力 {:3} : 节点 {:2} | {} 方向 | 大小: {:4e}".\
                          format( Fn , i // 3, n[i % 3], self.force["constraint"][i]),
                          file=file)
                    Fn += 1
    
    def calDeformGeo(self,deformSize="auto",refDeformSize=1):
        if deformSize == "auto":
            self.geo["deformSize"] = self.geo["L"] * 0.2 / np.abs(self.geo["DX"]).max() 
        else:
            self.geo["deformSize"] = deformSize
        self.geo["deformedX"] = self.geo["X"] + self.geo["DX"].reshape([-1,3]).T
        self.geo["deformedFigureX"] = self.geo["X"] + self.geo["DX"].reshape([-1,3]).T \
            * self.geo["deformSize"] * refDeformSize
    
    def resultFigure(self,deformSize='auto',refDeformSize=1,undeformedDisplay=True,nodeSize=1,
                     lineSize=1,forceDisplaySize=1,constraintDisplaySize=1,
                     opa=dict(node = 0.3, rod = 0.6, constraint = 1, force = 0.6),
                     undeOpa=dict(node = 0.3, rod = 0.6)):
        figs = []
        if not isinstance(opa,type({})):
            opa = dict(node = opa, rod = opa, constraint = opa, force = opa)
        if not isinstance(opa,type({})):
            undeOpa = dict(node = undeOpa, rod = undeOpa)
        if undeformedDisplay:
            # Add undeformed nodes
            figs += self.nodeFigure(nodeSize,undeOpa["node"],"undeformed_")
            # Add undeformed rods
            if self.type == "rod":
                figs += self.rodFigure(lineSize,undeOpa["rod"],"undeformed_")
        self.calDeformGeo(deformSize,refDeformSize)
        # Add nodes
        figs += self.nodeFigure(nodeSize, opa["node"],select="deformedFigureX")
        if self.type == "rod":
            # Add rods
            figs += self.rodFigure(lineSize,opa["rod"],select="deformedFigureX")
            # Add rod constraint
            figs += self.rodConstraintFigure(constraintDisplaySize,opa["constraint"]
                                             ,select="deformedFigureX")
            # Add rod node force
            figs += self.rodNodeForceFigure(forceDisplaySize,
                                            opa["force"],select="deformedFigureX")
            # Add constraint force
            figs += self.rodNodeForceFigure(forceDisplaySize,
                                            opa["force"],select="deformedFigureX",constraint=True)
        return figs
    
    def resultReport(self,text=True,figure=True,file=None,deformSize='auto',
                     refDeformSize=1,undeformedDisplay=True,nodeSize=1,
                     lineSize=1,forceDisplaySize=1,constraintDisplaySize=1,
                     opacity=dict(node = 0.3, rod = 0.6, constraint = 1, force = 0.6),
                     undeformedOpacity=dict(node = 0.3, rod = 0.6)):
        if text:
            self.resultText()
        if file != None:
            with open(file,'w+') as f:
                self.resultText(file=f)
            print("数据已保存至",file,"!")
        if figure:
            print("数据可视化:")
            fig = go.Figure(data=self.resultFigure(deformSize,refDeformSize,undeformedDisplay,nodeSize,
                     lineSize,forceDisplaySize,constraintDisplaySize,opacity,undeformedOpacity))
            # fig.update_layout(width = 1000, height = 800)
            fig.update_layout(title = "deform scale = "+"{:4e}".format(self.geo["deformSize"]))
            fig.show()
            
# TODO:绘图函数
    def nodeFigure(self,nodeSize,opa,name="",select="X"):
        figs = []
        for i in range(self.geo[select].shape[1]):
            x=self.geo[select][0,i]
            y=self.geo[select][1,i]
            z=self.geo[select][2,i]
            if select == "X":
                text = "<br>x:{:4e}  <br>y={:4e}  <br>z={:4e} <br>"\
                    .format(x,y,z)
            elif select == "deformedFigureX":
                DX = self.geo["DX"]
                text = ("<br>x:{:4e}  <br>y={:4e}  <br>z={:4e} <br>节点位移:<br> DX={:4e}"
                   " <br> DY={:4e} <br> DZ={:4e}<br>")\
                      .format(x,y,z,DX[i*3],DX[i*3+1],DX[i*3+2])
            figs.append(go.Scatter3d(x=[x],y=[y],z=[z],
                                mode='markers',
                                marker_color="blue",
                                marker=dict(size=10*nodeSize,opacity=opa,),
                                name=name+'node_'+str(i),
                                hovertemplate = name+'node_'+str(i)+text,
                                ))
        return figs
        
    def rodFigure(self,lineSize,opa,name="",select="X"):
        figs = []
        t = "rod"
        for i in range(self.unit[t]["ME"].shape[1]):
            n = self.unit[t]["ME"][:,i]
            if select == "X":
                text = "<br>节点:({:},{:}) <br>E:{:4e} <br>A:{:4e} <br>L:{:4e}<br>".\
                    format( n[0],n[1], self.unit["rod"]["E"][i],
                            self.unit["rod"]["A"][i],
                            self.unit["rod"]["L"][i])
            elif select == "deformedFigureX":
                F = self.unit[t]["nodeForce"][i]
                text = ("<br>节点:({:2},{:2}) <br>E:{:4e} <br>A:{:4e} <br>L:{:4e}<br>"
                        "节点{:2}内力:<br> X={:4e} <br> Y={:4e} <br> Z={:4e} <br> norm={:4e} <br>"
                         "节点{:2}内力:<br> X={:4e} <br> Y={:4e} <br> Z={:4e} <br> norm={:4e}").\
                    format( n[0],n[1], self.unit["rod"]["E"][i],
                            self.unit["rod"]["A"][i],
                            self.unit["rod"]["L"][i],
                            n[0],F[0],F[1],F[2],self.unit[t]["force"][i][0],
                            n[1],F[3],F[4],F[5],self.unit[t]["force"][i][1])
            figs.append(go.Scatter3d(x=[self.geo[select][0,n[0]],self.geo[select][0,n[1]]],
                                y=[self.geo[select][1,n[0]],self.geo[select][1,n[1]]],
                                z=[self.geo[select][2,n[0]],self.geo[select][2,n[1]]],
                                mode='lines',
                                line=dict(color='rgba(255, 0, 0, {:})'.format(opa),width=10*lineSize),
                                name=name+'rod_'+str(i),
                                hovertemplate=name+'rod_'+str(i)+text))
        return figs
    
    def rodConstraintFigure(self,constraintDisplaySize,opa,name="",select="X"):
        figs = []
        U = np.eye(3)
        DX = ["X","Y","Z"]
        for i in range(self.con["NR"].shape[0]):
            n = self.con["NR"][i] // 3
            a = self.con["NR"][i] % 3
            text = "<br>节点{:}<br>D{:} = {:} ".format(n, DX[a],self.con["DX"][i])
            figs.append(go.Cone(x=[self.geo[select][0,n]]*2,
                                y=[self.geo[select][1,n]]*2,
                                z=[self.geo[select][2,n]]*2,
                                u=[U[0,a],-U[0,a]],v=[U[1,a],-U[1,a]],w=[U[2,a],-U[2,a]],
                                colorscale='Greens',
                                sizemode="absolute",
                                sizeref=(0.4 * constraintDisplaySize),
                                anchor="tip",
                                showscale=False,
                                name=name+'constraint_'+str(i),
                                showlegend=True,
                                hovertemplate=name+'constraint_'+str(i)+text,
                                opacity=opa
                                ))
        return figs
    
    def rodNodeForceFigure(self,forceDisplaySize,opa,name="",select="X",constraint=False):
        figs = []
        if constraint:
            Ftype = "constraint"
            color = "magenta"
        else:
            Ftype = "node"
            color = "blues"
        F = self.force[Ftype].reshape(-1,3).T
        Fn = 0

        for i in range(F.shape[1]):
            if np.abs(F[:,i]).sum() != 0:
                text="<br>u:{:4e}<br>v:{:4e}<br>w:{:4e}<br>norm:{:4e}"\
                    .format(F[0,i],F[1,i],F[2,i],np.sum(F[:,i]**2)**0.5)
                figs.append(go.Cone(x=[self.geo[select][0,i]],
                                y=[self.geo[select][1,i]],
                                z=[self.geo[select][2,i]],
                                u=[F[0,i]],v=[F[1,i]],w=[F[2,i]],
                                colorscale=color,
                                sizemode="absolute",
                                sizeref= (0.8 * forceDisplaySize*(np.abs(F[:,i]).sum()/self.force["max"])**0.5),
                                anchor="tip",
                                showscale=False,
                                name=name+Ftype+"_force_"+str(Fn),
                                hovertemplate=name+Ftype+"_force_"+str(Fn)+text,
                                showlegend=True,
                                opacity=opa
                                ))
                Fn += 1
        return figs

# TODO:类对角矩阵存储
class diagMat:
    def __init__(self,n,L=False):
        self.M = [{'j' : i+1, 'v' : []} for i in range(n)]
        self.L = L
        
    def value(self,i,j):
        if i < j:
            if self.L: return 0
            i,j = j,i
        if j < self.M[i]['j']:
            return 0
        else:
            return self.M[i]['v'][j-self.M[i]['j']]
        
    def change(self,i,j,v):
        if i < j:
            if self.L: return None
            i,j = j,i
        if j < self.M[i]['j']:
            self.M[i]['j'], self.M[i]['v'] = j , [v] + ([0] * (self.M[i]['j'] - j -1)) + self.M[i]['v']
        else:
            self.M[i]['v'][j-self.M[i]['j']] = v
            
    def show(self):
        n = len(self.M)
        for i in range(n):
            s = ""
            for j in range(n):
                s += "{:4e},".format(self.value(i,j))
            print(s)

# TODO:cholesky算法
def cholesky(A,b):
    n = b.shape[0]
    d = np.zeros(n)
    x = np.zeros(n)
    l = diagMat(n,L = True)
    for i in range(n):
        for j in range(i):
            l.change(i,j,
                     (A.value(i,j)-sum([l.value(i,t) * d[t]* l.value(j, t) for t in\
                                        range(max(l.M[j]['j'],l.M[i]['j']), j)])) / d[j])
        l.change(i,i,1.0)
        d[i] = A.value(i,i) - sum([l.value(i,t)*l.value(i,t)*d[t] for t in range(l.M[i]['j'],i)])
    r = np.zeros(n)
    for i in range(n):
        r[i] = (b[i]-sum([l.value(i,j)*d[j]*r[j] for j in range(i)]))/d[i]
    for j in range(n-1,-1,-1):
        x[j] = r[j] - sum([l.value(i,j) * x[i] for i in range(j+1,n)])
    return x

# TODO:调用numpy库解方程
def mySolve(A,b):
    n = b.shape[0]
    K = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            K[i,j] = A.value(i,j)
    x = np.linalg.solve(K,b)
    return x

