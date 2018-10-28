package liblaries.neuralNetwork.learning;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.Linear;

public class GrowingNeuralGas extends LNetwork{
	
	private int[][] connections;							//[i][j] i-> neuron index, j-> index of connected neuron, value->life of connection, -2 did't exist, connection bidirectional, only update last occurence([max(x,y)][min{x,y)])
	private int[] connectionsNum;							//numbers of connection in each neuron
	
	private int maxNeurons=300;
	
	int maxLife;
	int addCycles;
	int cyclesToAdd;
	
//	float acumulatedError=0;
	
	public GrowingNeuralGas(int inputsNumber) {
		this(inputsNumber,300);
	}
	public GrowingNeuralGas(int inputsNumber, int maxNeurons) {
		this(inputsNumber,maxNeurons,4000,15);
	}
	public GrowingNeuralGas(int inputsNumber,int maxNeurons,int maxLife,int addCycles) {
		this.inputsNumber=inputsNumber;
		layersSize=new int[] {inputsNumber,2};
		layersNumber=1;
		setMaxNeurons(maxNeurons);
		this.maxLife=maxLife;
		this.addCycles=addCycles;
	}
	
	public void setMaxNeurons(int value) {					//cleans network conections and weights data
		if(!learning) {
			maxNeurons=value;
			connectionsNum=new int[value];
			connectionsNum[0]=connectionsNum[1]=1;
			connections=new int[value][];
			for(int i=0;i<value;i++) {
				connections[i]=new int[value-1];
				Arrays.fill(connections[i], -2);
			}
			connections[1][0]=0;
			weights=new float[1][maxNeurons][inputsNumber+1];
			deltaWeights=new float[1][maxNeurons][inputsNumber+1];
			outputs=new float[1][maxNeurons];
			outputFunction=new Linear();
			randomizeWeights();
		}else throw new NeuralException(NeuralException.learningInProgress);
	}
	
	public int getMaxNeurons() {
		return maxNeurons;
	}
	
	public void startLearning() {
		super.startLearning();
		if(cyclesToAdd<=0)
			cyclesToAdd=addCycles;
//		try {
//			if(log==null)
//				log=new PrintWriter(new FileOutputStream("a.txt"));
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
	}
	
	protected void lSimulate(int elemntNr, int start,int end,int layer, int threadID) {
		float[] inputData;
		if(layer>0)inputData=outputs[layer-1];
		else inputData=learningSequence[elemntNr].inputs;
		
		for(int i=start;i<end;i++) {
			outputs[layer][i]=0;
			for(int j=0;j<layersSize[layer];j++){
				float delta=weights[layer][i][j+1]-inputData[j];
				outputs[layer][i]+=delta*delta;
			}
			outputs[layer][i]=-(float)Math.sqrt(outputs[layer][i]);
			
			//output[layer][i]=function.function(output[layer][i]);
		}
	}
	public void lCountWeights(int elementNr, float n, float m) {
		int neurons=layersSize[1];
		int max=0;
		float maxValue=-Float.MAX_VALUE;
		int max2=0;
		float maxValue2=-Float.MAX_VALUE;
		for(int i=1;i<neurons;i++) {					//get closets nodes
			if(outputs[0][i]>maxValue) {
				max2=max;
				max=i;
				maxValue=outputs[0][i];
			} else if(outputs[0][i]>maxValue2) {
				max2=i;
				maxValue2=outputs[0][i];
			}
		}
		
		int winner=max;
		if(max<max2) {
			max=max2;
			max2=winner;
		}
		
//		acumulatedError+=outputs[0][winner];
		
		connections[max][max2]=-2;
		for(int i=0;i<winner;i++) {
			if(connections[i][winner]!=-2) {
				connections[i][winner]++;
				updateWeights(i,elementNr,n/2,m/2);							//TODO Set other n, m
				if(connections[i][winner]>maxLife)deleteConnection(i,winner);
			}
		}
		for(int i=winner+1;i<layersSize[1];i++) {
			if(connections[winner][i]!=-2) {
				connections[winner][i]++;
				updateWeights(i,elementNr,n/2,m/2);							//TODO Set other n, m
				if(connections[winner][i]>maxLife)deleteConnection(winner,i);
			}
		}
		connections[max][max2]=0;
		updateWeights(winner,elementNr,n,m);
		
		
	}
	//PrintWriter log=null;
	
	public void update(int cycle) {
		if(--cyclesToAdd==0&&layersSize[1]<maxNeurons-1) {
			cyclesToAdd=addCycles;
			
			float maxDistance=0;
			int neuron=0;
			int neuron2=0;
			for(int i=layersSize[1]-1;i>-1;i--) {			//get max connection lenght;
				for(int j=i-1;j>-1;j--) {
					if(connections[i][j]!=-2) {
						float distance=0;
						for(int k=0;k<layersSize[0];k++) {
							float delta=weights[0][i][k]-weights[0][j][k];
							distance+=delta*delta;
						}
						distance=(float)Math.sqrt(distance);
						if(distance>maxDistance) {
							maxDistance=distance;
							neuron=i;
							neuron2=j;
						}
					}
				}
			}
			
			int newIndex=layersSize[1]++;					//add neuron
			for(int i=0;i<inputsNumber;i++) {
				weights[0][newIndex][i]=(weights[0][neuron][i]+weights[0][neuron2][i])/2;
			}
			
			connections[neuron][neuron2]=-2;				//update connnections
			connections[newIndex][neuron]=0;
			connections[newIndex][neuron2]=0;
		}
//		if(cycle%10==0) {
//			//System.out.println(acumulatedError+" "+layersSize[1]);
//			log.println(Float.toString(acumulatedError));
//			log.flush();
//		}
//		acumulatedError=0;
	}
	private void deleteConnection(int a,int b) {
		System.out.println(a+" "+b);
		connections[a][b]=-2;
		if(--connectionsNum[a]==0) {
			System.arraycopy(weights, a+1, weights, a, layersSize[1]-a);
			for(int i=a;i<layersSize[1];i++) {
				System.arraycopy(connections[i+1], 0, connections[i], 0, layersSize[1]-a);
			}
		}
		if(--connectionsNum[b]==0) {
			System.arraycopy(weights, b+1, weights, b, layersSize[1]-b);
			for(int i=a;i<layersSize[1];i++) {
				System.arraycopy(connections[i+1], 0, connections[i], 0, layersSize[1]-b);
			}
		}

	}
	private void updateWeights(int neuron,int elementNr,float n,float m) {
		for(int k=1;k<inputsNumber;k++) {
			float delta=m*deltaWeights[0][neuron][k]+n*(learningSequence[elementNr].inputs[k-1]-weights[0][neuron][k]);
			weights[0][neuron][k]+=delta;
			deltaWeights[0][neuron][k]=delta;
		}
	}
}