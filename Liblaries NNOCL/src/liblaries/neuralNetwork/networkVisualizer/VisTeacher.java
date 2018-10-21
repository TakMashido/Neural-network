package liblaries.neuralNetwork.networkVisualizer;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.learning.Teacher;

public class VisTeacher extends Teacher {
	
	private int indexN=0;
	private int indexM=0;
	
	public VisTeacher(){
		super();
	}
	public VisTeacher(int cyclesNumber) {
		super(cyclesNumber);
	}
	public VisTeacher(float N,float M,int cyclesNumber){
		super(N,M,cyclesNumber);
	}
	public VisTeacher(float[] N,float[] M,int cyclesNumber){
		super(N,M,cyclesNumber);
	}
	public VisTeacher(float[] N,int[] changeN,float[] M,int[] changeM,int cyclesNumber){
		super(N,changeN,M,changeM,cyclesNumber);
	}
	public VisTeacher(String nnName,String lsName,int cyclesNumber,float N,float M) throws IOException, NeuralException{
		super(nnName, lsName, cyclesNumber, N, M);
	}
	
	public void reset() {
		elementNr=actualCycle=indexM=indexN=0;
	}
	public void bigTick() {
		if(elementNr==0)tick();
		while(elementNr!=0)
			tick();
	}
	public void tick() {
		try {
			network.action(elementNr, n, m);
		} catch (BrokenBarrierException|InterruptedException e) {e.printStackTrace();}
		
		if(++elementNr==network.getLSLenght()) {
			network.update(++actualCycle);
			
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
			elementNr=0;
		}
	}
}