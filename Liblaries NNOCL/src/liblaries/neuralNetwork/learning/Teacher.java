package liblaries.neuralNetwork.learning;

import java.io.IOException;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;

import liblaries.neuralNetwork.errors.NeuralException;

public class Teacher{
	private LNetwork network;
	public float n=.2f;
	public float m=.13f;
	public int cycleNumber=10000;
	
	public float[] dn;														//n and m change values
	public float[] dm;
	public int[] zn;
	public int[] zm;														//index of cycles of n or m change
	
	public Random random=new Random();
	
	private int elementNr;													//number of actual simulate element from LS
	
	private int actualCycle=-2;											//-1: checking LS, -2: learning end, other: actual cycle of learn
	
	public Teacher(){
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new int[]{cycleNumber};
		zm=new int[]{cycleNumber};
	}
	public Teacher(int cyclesNumber) {
		this();
		this.cycleNumber=cyclesNumber;
	}
	public Teacher(float N,float M,int cyclesNumber){
		n=N;
		m=M;
		cycleNumber=cyclesNumber;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new int[]{cyclesNumber};
		zm=new int[]{cyclesNumber};
	}
	public Teacher(float[] N,float[] M,int cyclesNumber){
		dn=N;
		dm=M;
		cycleNumber=cyclesNumber;
		
		n=dn[0];
		m=dm[0];
		
		zn=new int[dn.length];
		zm=new int[dm.length];
		
		int remaining=cycleNumber;
		int delta;
		for(int i=dn.length-1;i>0;i--){
			delta=remaining/i;
			
			zn[i]=remaining;
			remaining-=delta;
		}
		
		remaining=0;
		for(int i=dm.length-1;i>0;i--){
			delta=remaining/i;
			
			zm[i]=remaining;
			remaining-=delta;
		}
	}
	public Teacher(float[] N,int[] changeN,float[] M,int[] changeM,int cyclesNumber){
		dn=N;
		dm=M;
		
		zn=changeN;
		zm=changeM;
		
		cycleNumber=cyclesNumber;
	}
	public Teacher(String nnName,String lsName,int cyclesNumber,float N,float M) throws IOException, NeuralException{
		network=FileL.readLNetwork(nnName);
		network.setLS(FileL.readLS(lsName));
		checkTS();
				
		cycleNumber=cyclesNumber;
		n=N;
		m=M;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new int[]{cyclesNumber};
		zm=new int[]{cyclesNumber};
	}
	
	public void setLinearN(float nMax, float nMin,int stepsNumber) {
		float delta=(nMax-nMin)/(stepsNumber);
		float deltaIndex=cycleNumber/(float)stepsNumber;
		dn=new float[stepsNumber];
		zn=new int[stepsNumber];
		for(int i=0;i<dn.length;i++) {
			dn[i]=nMax-i*delta;
			zn[i]=(int)deltaIndex*i;
		}
	}
	public void setLinearM(float mMax, float mMin,int stepsNumber) {
		float delta=(mMax-mMin)/(stepsNumber);
		float deltaIndex=cycleNumber/(float)stepsNumber;
		dm=new float[stepsNumber];
		zm=new int[stepsNumber];
		for(int i=0;i<dm.length;i++) {
			dm[i]=mMax-i*delta;
			zm[i]=(int)deltaIndex*i;
		}
	}
	
	public void setNetwork(LNetwork network) {
		if(actualCycle==-2) {
			this.network=network;
		}else
			throw new NeuralException(3);
	}
	
	public LNetwork getNetwork() {
		return network;
	}
	
	
	public LNetwork teach() throws NeuralException{
		network.startLearning();
		
		actualCycle=-1;
		checkTS();
		
		int elementsCount=network.getLSLenght();
		
		int indexN=0;												//index of next n, m in array
		int indexM=0;
		
		n=dn[0];
		m=dm[0];
		
		for(actualCycle=0;actualCycle<cycleNumber;actualCycle++){
			for(elementNr=0;elementNr<elementsCount;elementNr++){
				try {
					network.action(elementNr, n, m);
				} catch (InterruptedException | BrokenBarrierException e) {
					NeuralException exception=new NeuralException(NeuralException.multiThreadingError);
					exception.setStackTrace(e.getStackTrace());
					throw exception;
				}
			}
			network.update(actualCycle);
			
			if(actualCycle%100==0){
				network.mixLS(random);
			}
			
			if(indexN<zn.length&&actualCycle==zn[indexN]){							//Change values of n m
				n=dn[indexN];
				indexN++;
			}
			if(indexM<zm.length&&actualCycle==zm[indexM]){
				m=dn[indexM];
				indexM++;
			}
		}
		
		network.endLearning();
		actualCycle=-2;
		return network;
	}
	private void checkTS() throws NeuralException{
		int inputNumber=network.getInputNumber();
		int outputNumber=network.getOutputNumber();
		LearningSequence[] ci¹gUcz¹cy=network.getLS();
		
		for(LearningSequence CU:ci¹gUcz¹cy){
			if(CU.inputs.length!=inputNumber)
				throw new NeuralException(NeuralException.invalidInputSize);
			if(CU.outputs!=null&&CU.outputs.length!=outputNumber)
				throw new NeuralException(NeuralException.invalidOutputSize);
		}
	}
	public final long getCycle() {
		return actualCycle;
	}
}