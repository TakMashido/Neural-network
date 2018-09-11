package liblaries.neuralNetwork.learning;

import java.util.function.BiFunction;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.Function;

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
	
	public KohonenNetwork(){}
	public KohonenNetwork(float[][][] weights, Function function) {
		super(weights,function);
	}
	public KohonenNetwork(float[][][] weights, Function function, boolean initializeOpenCL) {
		super(weights,function);
		if(initializeOpenCL) initializeOpenCL();
	}
	public KohonenNetwork(int inputsNumber,int[] layersSize, Function function) {
		super(inputsNumber,layersSize,function);
	}
	public KohonenNetwork(int inputsNumber,int[] layersSize, Function function, boolean initializeOpenCL) {
		super(inputsNumber,layersSize,function);
		if(initializeOpenCL) initializeOpenCL();
	}
	protected void createLNetwork(int inputsNumber,int[] layersSize,Function function){
		super.createLNetwork(inputsNumber, layersSize, function);
		for(float[][] layer:weights)
			for(float[] neuron:layer) {
				neuron[0]=0;
				normalizeData(neuron);
			}
	}
	
	public void normalizeData(float[] data) {
		float sum=0;
		for(int i=0;i<data.length;i++)
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
	public void setLS(LearningSequence[] ls) {
		for(LearningSequence lsEl:ls) {
			normalizeData(lsEl.inputs);
		}
		
		super.setLS(ls);
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
	
//	public void prepareCLMem() {
//		super.prepareCLMem();
//		winIndex=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float, Pointer.to(new float[] {-1}), null);
//	}
//	protected void clearCLMem() {
//		super.clearCLMem();
//		CL.clReleaseMemObject(winIndex);
//	}
	int cycleNumber;
	public void countWeights(int elementNr, float n, float m) {
		//if(elementNr%100==0)
		//	System.out.println(Arrays.toString(weights[0][0])+"\n"+Arrays.toString(deltaWeights[0][0])+"\n");
		//if(elementNr==658)
		//	System.out.println(Arrays.toString(weights[0][0])+"\n"+Arrays.toString(deltaWeights[0][0])+"\n");
		for(int i=0;i<weights.length;i++) {
			int maxIndex=0;
			for(int j=1;j<weights[i].length;j++) 
				if(output[i][j]>output[i][j-1])maxIndex=j;
			
			for(int j=0;j<weights[i].length;j++) {
				float h=distanceFunction.apply(j, maxIndex);
//				if(!Float.isFinite(h)) {
//					System.out.println("h: "+weights[i][j][0]+" "+h+" "+Arrays.toString(learningSequence[elementNr].inputs));
//					System.exit(-1);
//				}
				//System.out.println(h);
				if(h!=0)
					for(int k=1;k<weights[i][j].length;k++) {
//						if(j==0&k==9&elementNr==653)
//							System.out.println(weights[i][j][k]);
						float delta=m*deltaWeights[i][j][k]+n*h*((i==0?learningSequence[elementNr].inputs[k-1]:output[i-1][k-1])-weights[i][j][k]);
//						if(!Float.isFinite(delta)) {
//							System.out.println("d: "+m+" "+deltaWeights[i][j][k]+" "+n+" "+h+" "+((i==0?learningSequence[elementNr].inputs[k-1]:output[i-1][k-1])+" "+weights[i][j][k]));
//							System.exit(-1);
//						}
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
			}
		}
	}
	public void update(int cycleNumber) {
		if(dIndex!=zd.length&&cycleNumber<zd[dIndex])
			maxDistance=dd[dIndex++];
		this.cycleNumber=cycleNumber;
	}
}