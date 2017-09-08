package liblaries.neuralNetwork.learning;

import java.io.IOException;
import java.util.Random;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.errors.NeuralException;

public class Teacher{
	private LNetwork network;
	public float n;
	public float m;
	public long cycleNumber;
	
	public float[] dn;														//zmienia wartoœæ n i m
	public float[] dm;
	public long[] zn;
	public long[] zm;														//przy o ilu cyklach zmienia siê n i m
	
	public Random random=new Random();
	
	private int NrElementu;													//numer aktu³alnie rozwarzanego elementu z ci¹gu ucz¹cego
	
	private long aktu³alnyCykl=-2;											//-1: checking LS, -2: learning end, other: actual cycle of learn
	
	public Teacher(){
		n=0.2f;
		m=0.13f;
		cycleNumber=10000;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{10000};
		zm=new long[]{10000};
	}
	public Teacher(float N,float M,long cyclesNumber){
		n=N;
		m=M;
		cycleNumber=cyclesNumber;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{cyclesNumber};
		zm=new long[]{cyclesNumber};
	}
	public Teacher(float[] N,float[] M,long cyclesNumber){
		dn=N;
		dm=M;
		cycleNumber=cyclesNumber;
		
		n=dn[0];
		m=dm[0];
		
		zn=new long[dn.length];
		zm=new long[dm.length];
		
		long remaining=cycleNumber;						//okreœla ile cyk. ucz.pozosta³o do podzia³u
		long delta;
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
	public Teacher(float[] N,long[] zmianaN,float[] M,long[] zmianaM,long LiczbaCykli){
		dn=N;
		dm=M;
		
		zn=zmianaN;
		zm=zmianaM;
		
		cycleNumber=LiczbaCykli;
	}
	public Teacher(String NazwaSN,String NazwaCU,long LiczbaCykli,float N,float M) throws FileVersionException, IOException, NeuralException{
		network=FileL.readLNetwork(NazwaSN);
		network.setLS(FileL.readLS(NazwaCU));
		checkTS();
				
		cycleNumber=LiczbaCykli;
		n=N;
		m=M;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{LiczbaCykli};
		zm=new long[]{LiczbaCykli};
	}
	
	public void setNetwork(LNetwork network) {
		if(aktu³alnyCykl==-2) {
			this.network=network;
		}else
			throw new NeuralException(3);
	}
	
	public LNetwork teach() throws NeuralException{
		network.startLearning();
		
		aktu³alnyCykl=-1;
		checkTS();
		
		int nrElement=network.getLSLenght();
		
		int indexN=0;												//index nastêpnego n, m w tablicy
		int indexM=0;
		
		for(aktu³alnyCykl=0;aktu³alnyCykl<cycleNumber;aktu³alnyCykl++){
			for(NrElementu=0;NrElementu<nrElement;NrElementu++){
				//System.out.println("symulating network");
				network.LSimulateNetwork(NrElementu);						//zasymulowanie dzia³ania sieci
				
				//System.out.println("counting error");
				network.coutError(NrElementu);
				
				//System.out.println("countion weights");
				network.countWeights(NrElementu,n,m);
			}
			if(aktu³alnyCykl%100==0){								//Randomizacja CU
				network.mixLS(random);
			}
			
			if(aktu³alnyCykl==zn[indexN]){							//Zmiana wartoœci n i m zgodnie z tablicami
				indexN++;
				n=dn[indexN];
			}
			if(aktu³alnyCykl==zm[indexM]){
				indexM++;
				m=dn[indexM];
			}
			
			//try{													//XXX SN "och³adzacz" procka
			//	Thread.sleep(500);
			//}catch(InterruptedException e){
			//	e.printStackTrace();
			//}
		}
		
		network.endLearning();
		aktu³alnyCykl=-2;
		return network;
	}
	private void checkTS() throws NeuralException{					//Sprawdza czy CU odpowiada SN
		int inputNumber=network.getInputNumber();
		int outputNumber=network.getOutputNumber();
		LearningSeqence[] ci¹gUcz¹cy=network.getCU();
		
		for(LearningSeqence CU:ci¹gUcz¹cy){
			if(CU.inputs.length!=inputNumber){
				throw new NeuralException(2);
			}
			if(CU.outputs.length!=outputNumber){
				throw new NeuralException(2);
			}
		}
	}
	public final long getCykl() {
		return aktu³alnyCykl;
	}
}