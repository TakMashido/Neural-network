package liblaries.neuralNetwork.learning;

import java.io.IOException;
import java.util.Random;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.errors.NeuralException;

public class Teacher{
	public LNetwork network=new LNetwork();
	public float n;
	public float m;
	public long cycleNumber;
	
	public float[] dn;														//zmienia wartość n i m
	public float[] dm;
	public long[] zn;
	public long[] zm;														//przy o ilu cyklach zmienia się n i m
	
	public Random random=new Random();
	
	private int NrElementu;													//numer aktułalnie rozwarzanego elementu z ciągu uczącego
	
	private long aktułalnyCykl=-2;											//-1: sprawdzanieCU, -2: uczenie zakończone, inne: aktułalny cykl uczenia
	
	public Teacher(){
		n=0.2f;
		m=0.13f;
		cycleNumber=10000;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{10000};
		zm=new long[]{10000};
	}
	public Teacher(float N,float M,long LiczbaCykli){
		n=N;
		m=M;
		cycleNumber=LiczbaCykli;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{LiczbaCykli};
		zm=new long[]{LiczbaCykli};
	}
	public Teacher(float[] N,float[] M,long LiczbaCykli){
		dn=N;
		dm=M;
		cycleNumber=LiczbaCykli;
		
		n=dn[0];
		m=dm[0];
		
		zn=new long[dn.length];
		zm=new long[dm.length];
		
		long pozostało=cycleNumber;						//określa ile cyk. ucz.pozostało do podziału
		long delta;
		for(int i=dn.length-1;i>0;i--){
			delta=pozostało/i;
			
			zn[i]=pozostało;
			pozostało-=delta;
		}
		
		pozostało=0;
		for(int i=dm.length-1;i>0;i--){
			delta=pozostało/i;
			
			zm[i]=pozostało;
			pozostało-=delta;
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
		network=FileN.readLNetwork(NazwaSN);
		network.setLS(FileN.readLS(NazwaCU));
		checkTS();
				
		cycleNumber=LiczbaCykli;
		n=N;
		m=M;
		
		dn=new float[]{n};
		dm=new float[]{m};
		
		zn=new long[]{LiczbaCykli};
		zm=new long[]{LiczbaCykli};
	}
	
	public LNetwork teach() throws NeuralException{
		network.startLearning();
		
		aktułalnyCykl=-1;
		checkTS();
		
		int nrElement=network.getCULenght();
		
		int indexN=0;												//index następnego n, m w tablicy
		int indexM=0;
		
		for(aktułalnyCykl=0;aktułalnyCykl<cycleNumber;aktułalnyCykl++){
			for(NrElementu=0;NrElementu<nrElement;NrElementu++){
				//System.out.println("symulating network");
				network.LSimulateNetwork(NrElementu);						//zasymulowanie działania sieci
				
				//System.out.println("counting error");
				network.coutError(NrElementu);
				
				//System.out.println("countion weights");
				network.countWeights(NrElementu,n,m);
			}
			if(aktułalnyCykl%100==0){								//Randomizacja CU
				network.mixLS(random);
			}
			
			if(aktułalnyCykl==zn[indexN]){							//Zmiana wartości n i m zgodnie z tablicami
				indexN++;
				n=dn[indexN];
			}
			if(aktułalnyCykl==zm[indexM]){
				indexM++;
				m=dn[indexM];
			}
			
			//try{													//XXX SN "ochładzacz" procka
			//	Thread.sleep(500);
			//}catch(InterruptedException e){
			//	e.printStackTrace();
			//}
		}
		
		network.endLearning();
		aktułalnyCykl=-2;
		return network;
	}
	private void checkTS() throws NeuralException{					//Sprawdza czy CU odpowiada SN
		int inputNumber=network.getInputNumber();
		int outputNumber=network.getOutputNumber();
		LearningSeqence[] ciągUczący=network.getCU();
		
		for(LearningSeqence CU:ciągUczący){
			if(CU.inputs.length!=inputNumber){
				throw new NeuralException(2);
			}
			if(CU.outputs.length!=outputNumber){
				throw new NeuralException(2);
			}
		}
	}
	public final long getCykl() {
		return aktułalnyCykl;
	}
}