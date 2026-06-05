import sys
import numpy as np
import cv2
cv2.setLogLevel(0)
from PyQt6.QtWidgets import QApplication,QWidget
from PyQt6.QtGui import QPainter,QPen,QImage,QColor
from PyQt6.QtCore import QTimer,Qt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
class FourierDrawer(QWidget):
	def __init__(self,FileName:str):
		super().__init__()
		self.FileName=FileName
		self.img=cv2.imread(f"{self.FileName}.png")
		if self.img is None:
			print(f"Error in Loading Image ({self.FileName}.png)")
			exit(0)
		self.setWindowTitle(self.FileName)
		self.resize(600,600)
		self.img=cv2.resize(self.img,dsize=(600,600))
		gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		edges=cv2.Canny(gray,50,200)
		self.contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		canvas=np.zeros_like(gray)
		cv2.drawContours(canvas,self.contours,-1,255,1)
		self.points_complex=[]
		self.fourier=[]
		self.freqs=[]
		self.time=0
		self.prev_x=0
		self.prev_y=0
		self.Index=0
		self.path=[]
		self.currentNumberofPoints=-1
		self.isSettingsApplied=False
		self.NumberofContours=len(self.contours)
		self.canvas=QImage(self.width(),self.height(),QImage.Format.Format_RGB32)
		self.canvas.fill(Qt.GlobalColor.black)
		self.timer=QTimer()
		self.timer.timeout.connect(self.update_animation)
		self.timer.start(1)
	def update_animation(self):
		if self.img is None:
			return
		if self.time==self.currentNumberofPoints:
			self.time=0
			self.Index+=1
			self.isSettingsApplied=False
		if (not self.isSettingsApplied) and (self.Index<self.NumberofContours):
			contour=self.contours[self.Index]
			points=contour.reshape(-1,2)
			self.points_complex=points[:,0]+1j*points[:,1]
			self.fourier=np.fft.fft(self.points_complex)
			self.freqs=np.fft.fftfreq(len(self.points_complex))
			indices=np.argsort(np.abs(self.fourier))[::-1]
			self.fourier=self.fourier[indices]
			self.freqs=self.freqs[indices]
			self.currentNumberofPoints=len(self.fourier)
			self.prev_x=0
			self.prev_y=0
			self.path=[]
			self.isSettingsApplied=True
		self.time+=1
		self.update()
	def paintEvent(self,event):
		if self.img is None:
			return
		if self.Index==self.NumberofContours:
			return
		painter=QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		painter.drawImage(0,0,self.canvas)
		canvas_painter=QPainter(self.canvas)
		canvas_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		if len(self.freqs)!=0:
			freq=self.freqs[0]
			phase=np.angle(self.fourier[0])
			radius=abs(self.fourier[0])/len(self.points_complex)
			x,y=radius*np.cos(2*np.pi*freq*self.time+phase),radius*np.sin(2*np.pi*freq*self.time+phase)
			for i in range(1,min(90,len(self.fourier))):
				self.prev_x,self.prev_y=x,y
				freq=self.freqs[i]
				radius=abs(self.fourier[i])/len(self.points_complex)
				phase=np.angle(self.fourier[i])
				x+=radius*np.cos(2*np.pi*freq*self.time+phase)
				y+=radius*np.sin(2*np.pi*freq*self.time+phase)
				painter.setPen(QPen(Qt.GlobalColor.darkGray,1))
				painter.drawEllipse(
					int(self.prev_x-radius),
					int(self.prev_y-radius),
					int(2*radius),
					int(2*radius)
				)
				painter.drawLine(
					int(self.prev_x),
					int(self.prev_y),
					int(x),
					int(y)
				)
			self.path.append((x,y))
			canvas_painter.setPen(QPen(Qt.GlobalColor.blue,1))
			for i in range(1,len(self.path)):
				x1,y1=self.path[i-1]
				x2,y2=self.path[i]
				x1=int(x1)
				y1=int(y1)
				x2=int(x2)
				y2=int(y2)
				canvas_painter.setPen(QPen(QColor(self.img[y1][x1][2],self.img[y1][x1][1],self.img[y1][x1][0]),1))
				canvas_painter.drawLine(x1,y1,x2,y2)
FileName=input("Enter Image name : ")
app=QApplication(sys.argv)
window=FourierDrawer(FileName)
window.show()
sys.exit(app.exec())