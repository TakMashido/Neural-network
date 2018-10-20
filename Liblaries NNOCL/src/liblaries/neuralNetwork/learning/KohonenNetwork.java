package liblaries.neuralNetwork.learning;

import java.util.function.BiFunction;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.OutputFunction;

public class KohonenNetwork extends LNetwork {
	public final BiFunction<Integer,Integer,Float> linear1DFunction=new BiFunction<Integer,Integer,Float>(){
		public Float apply(Integer t, Integer u) {
			int distance=t-u;
			if(distance<0)distance*=-1;
			if(distance>maxDistance)return 0f;
			return (maxDistance-distance)/(float)maxDistance;
		}
	};
	
	protected int maxDistance=3;				//max distance beetwen win neuron and learned neurons
	protected int dIndex=0;
	protected int[] dd=new int[0];				//Values of distance
	protected int[] zd=new int[0];				//Index of distanceChange
	protected BiFunction<Integer,Integer,Float> distanceFunction=linear1DFunction;
	
	int[] maxIndex;								//Index and value of neuron of each part of network. Used in multiThreading
	float[] maxValue;
	
	public KohonenNetwork(float[][][] weights, OutputFunction outputFunction) {
		super(weights,outputFunction);
		init();
	}
	public KohonenNetwork(float[][][] weights, OutputFunction outputFunction, boolean initializeOpenCL) {
		super(weights,outputFunction);
		if(initializeOpenCL) initializeOpenCL();
		init();
	}
	public KohonenNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction) {
		super(inputsNumber,layersSize,outputFunction);
		for(int i=0;i<weights.length;i++) {
			for(int j=0;j<weights[i].length;j++) {
				normalizeData(weights[i][j]);
			}
		}
		init();
	}
	public KohonenNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction, boolean initializeOpenCL) {
		super(inputsNumber,layersSize,outputFunction);
		if(initializeOpenCL) initializeOpenCL();
		init();
	}
	protected void init() {
		//supportsMultiThreading=false;
	}
	protected void createLNetwork(int inputsNumber,int[] layersSize,OutputFunction outputFunction){
		super.createLNetwork(inputsNumber, layersSize, outputFunction);
		for(float[][] layer:weights)
			for(float[] neuron:layer) {
				neuron[0]=0;
				//normalizeData(neuron);
			}
	}
	
	public void normalizeData(float[] data) {
		float sum=data[0]*data[0];
		for(int i=1;i<data.length;i++)
			sum+=data[i]*data[i];
		if(sum!=0) {
			sum=(float)Math.sqrt(sum);
			for(int i=0;i<data.length;i++)
				data[i]/=sum;
		}
	}
	
	public void setDistances(int[] dd,int[] zd) {
		this.dd=dd;
		this.zd=zd;
		dIndex=0;
		maxDistance=dd[0];
	}
	public void setDistance(int max,int min,int cyclesPerChange) {
		if(min==0)min=1;
		min--;
		dd=new int[max-min];
		zd=new int[dd.length];
		dIndex=0;
		maxDistance=max;
		
		for(int i=0;i<dd.length;i++){
			dd[i]=max--;
			zd[i]=cyclesPerChange*i;
		}
	}
	public void initializeOpenCL() {
		throw new NeuralException(NeuralException.notSupportOpenCL);
	}
	
	public void setMaxDistance(int maxDistance) {
		this.maxDistance=maxDistance;
	}
	public void setDistanceFunction(BiFunction<Integer,Integer,Float> function) {
		distanceFunction=function;
	}
	public void setLiner1DFunction() {
		distanceFunction=linear1DFunction;
	}
	public void setLinear2DSquereFunction(int rowsNumber) {
		distanceFunction=new BiFunction<Integer,Integer,Float>(){
			public Float apply(Integer t, Integer u) {
				int dy=t/rowsNumber-u/rowsNumber;
				int dx=t%rowsNumber-u%rowsNumber;
				if(dy<0)dy*=-1;
				if(dx<0)dx*=-1;
				
				int distance=dy+dx;
				if(distance>maxDistance)return 0f;
				return (maxDistance-distance)/(float)maxDistance;
			}
		};
	}
	public void setLinear2DEuclideanFunction(int rowsNumber) {
		distanceFunction=new BiFunction<Integer,Integer,Float>(){
			public Float apply(Integer t, Integer u) {
				int dy=t/rowsNumber-u/rowsNumber;
				int dx=t%rowsNumber-u%rowsNumber;
				
				int distance=dy*dy+dx*dx;
				distance=(int)Math.sqrt(distance);
				if(distance>maxDistance)return 0f;
				
				return (maxDistance-distance)/(float)maxDistance;
			}
		};
	}
	
	public void setThreadsNumber(int threads) {
		supportsMultiThreading=true;
		super.setThreadsNumber(threads);
		maxIndex=new int[threadsNumber];
		maxValue=new float[threadsNumber];
	}
	
	public byte getSimMethodID() {
		return 1;
	}
	
//	public void prepareCLMem() {
//		super.prepareCLMem();
//		winIndex=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float, Pointer.to(new float[] {-1}), null);
//	}
//	protected void clearCLMem() {
//		super.clearCLMem();
//		CL.clReleaseMemObject(winIndex);
//	}
	protected void lSimulate(int elementNr) {
		float[] inputData=learningSequence[elementNr].inputs;
		
		for(int i=0;i<layersNumber;i++) {
			int index=0;
			if(i>0)inputData=outputs[i-1];
			
			for(int j=0;j<layersSize[i];j++) {
				outputs[i][j]=0;
				for(int k=1;k<weights[i][j].length;k++){
					float delta=weights[i][j][k]-inputData[k-1];
					outputs[i][j]+=delta*delta;
					//output[i][j]=Math.fma(delta, delta, output[i][j]);
				}
				outputs[i][j]=-(float)Math.sqrt(outputs[i][j]);
				
				//output[layer][i]=function.function(output[layer][i]);
				
				if(outputs[i][j]>outputs[i][index])index=j;
			}
		}
	}
	protected void lSimulate(int elementNr, int start,int end,int layer,int threadID) {
		float[] inputData;
		if(layer>0)inputData=outputs[layer-1];
		else inputData=learningSequence[elementNr].inputs;
		
		int index=0;
		for(int i=start;i<end;i++) {
			outputs[layer][i]=weights[layer][i][0];
			for(int j=1;j<weights[layer][i].length;j++){
				float delta=weights[layer][i][j]-inputData[j-1];
				outputs[layer][i]+=delta*delta;
			}
			outputs[layer][i]=-(float)Math.sqrt(outputs[layer][i]);
			
			//output[layer][i]=function.function(output[layer][i]);
			
			if(outputs[layer][i]>outputs[layer][index])index=i;
		}
		maxIndex[threadID]=index;
		maxValue[threadID]=outputs[layer][index];
	}
	public void lCountWeights(int elementNr, float n, float m) {
		for(int i=0;i<weights.length;i++) {
			int maxIndex=0;
			for(int j=1;j<weights[i].length;j++) 
				if(outputs[i][j]>outputs[i][maxIndex])maxIndex=j;
			
			for(int j=0;j<weights[i].length;j++) {
				float h=distanceFunction.apply(j, maxIndex);
				if(h!=0)
					for(int k=1;k<weights[i][j].length;k++) {
						float delta=m*deltaWeights[i][j][k]+n*h*((i==0?learningSequence[elementNr].inputs[k-1]:outputs[i-1][k-1])-weights[i][j][k]);
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
			}
		}
	}
	public void lCountWeights(int elementNr, float n, float m, int start, int end, int layer, int threadID) {
		int index=0;
		for(int j=0;j<maxIndex.length;j++) 
			if(maxValue[j]>maxValue[index])index=j;
		
		index=maxIndex[index];;
		
		for(int j=start;j<end;j++) {
			float h=distanceFunction.apply(j, index);
			if(h!=0)
				for(int k=1;k<weights[layer][j].length;k++) {
					float delta=m*deltaWeights[layer][j][k]+n*h*((layer==0?learningSequence[elementNr].inputs[k-1]:outputs[layer-1][k-1])-weights[layer][j][k]);
					weights[layer][j][k]+=delta;
					deltaWeights[layer][j][k]=delta;
				}
		}
			
	}
	private int cycleNumber;
	private int lastCycle;
	public void update(int cycle) {
		int delta=cycle-lastCycle;
		if(delta<0)delta=cycle;
		this.cycleNumber+=delta;
		if(dIndex!=zd.length&&cycleNumber<zd[dIndex])
			maxDistance=dd[dIndex++];
	}
}