package liblaries.neuralNetwork.learning;

import java.util.HashMap;
import java.util.Random;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.Function;

public abstract class LNetwork{
	protected float[][][] weights;												//input layer have  ID=0	[a][b][c] a->layer, b->neuron, c->connection
	protected float[][][] deltaWeights;											//							[a][b][c]
	protected float[][] error;													//							[a][b]
	protected float[][] output;													//							[a][b]
	protected int inputsNumber;
	
	protected LearningSequence[] learningSequence;
	protected Function function;
	
	int[] layersSize;															//[0] inputNumber, [1] neurons in layer 0, [2] neurons in layer 1, ...
	int layersNumber;
	
	protected boolean openCLLoaded=false;
	protected cl_context context;
	protected cl_command_queue commandQueue;
	
	protected cl_program program;
	protected cl_kernel simulateKernel;
	protected cl_kernel outputLayerErrorKernel;
	protected cl_kernel calculateErrorKernel;
	protected cl_kernel calculateWeightsKernel;
	
	protected cl_mem[] weightsCL;							//[a] a->layer
	protected cl_mem[] deltaWeightsCL;						//[a]
	protected cl_mem[] outputsCL;							//[a]			//with bias
	protected cl_mem[] errorCL;								//[a]			//without bias
	
	protected int threadsNumber=1;
	//protected ExecutorService executor=Executors.newCachedThreadPool();
	
	protected boolean learning=false;
	
	public LNetwork(){addOpenCLProgram();}
	public LNetwork(float[][][] weights, Function function) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.function=function;
		addOpenCLProgram();
	}
	public LNetwork(float[][][] weights, Function function, boolean initializeOpenCL) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.function=function;
		
		addOpenCLProgram();
		if(initializeOpenCL)
			initializeOpenCL();
	}
	public LNetwork(int inputsNumber,int[] layersSize, Function function) {
		createLNetwork(inputsNumber, layersSize, function);
		prepareData();
		addOpenCLProgram();
	}
	public LNetwork(int inputsNumber,int[] layersSize, Function function, boolean initializeOpenCL) {
		createLNetwork(inputsNumber, layersSize, function);
		addOpenCLProgram();
		if(initializeOpenCL) {
			initializeOpenCL();
		}
		else
			prepareData();
	}
	protected void addOpenCLProgram() {
		openCLProgram.put("simulate",
			   "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
			 + "	int neuron=get_global_id(0);"
			 + "	int index=neuron*connectionsNumber;"
			 + "	neuron++;"								//offset of worksize in not supported by openCL yet(neuron[0]==bias)
			 + "	"
			 + "	float value=0;"
			 + "	for(int i=0;i<connectionsNumber;i++){"
			 + "		value+=weights[index+i]*input[i];"
			 + "	}"
			 + "	"
			 + "	output[neuron]=outputFunction(value);"
			 + "}");
	}
	protected void createLNetwork(int inputsNumber,int[] layersSize,Function function){
		this.inputsNumber=inputsNumber;
		
		Random random=new Random();
		
		weights=new float[layersSize.length][][];
		for(int i=0;i<layersSize.length;i++){
			int connectionsNumber=(i==0?inputsNumber:layersSize[i-1])+1;
			weights[i]=new float[layersSize[i]][connectionsNumber];
			
			float maxWeighth=1/(float)(layersSize[i]/20+1)+0.000000000000000001f;
			
			for(int j=0;j<layersSize[i];j++){
				for(int k=0;k<connectionsNumber;k++){
					while(true){
						float weight=random.nextFloat();
						if(weight!=0){
							weights[i][j][k]=weight%maxWeighth;
							break;
						}
					}
				}
			}
		}
		this.function=function;
	}
	protected void prepareData() {
		deltaWeights=new float[weights.length][][];
		output=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			deltaWeights[i]=new float[weights[i].length][];
			output[i]=new float[weights[i].length];
			
			for(int j=0;j<weights[i].length;j++) {
				deltaWeights[i][j]=new float[weights[i][j].length];
			}
		}
		
		layersNumber=weights.length;
		
		layersSize=new int[weights.length+1];
		layersSize[0]=inputsNumber;
		for(int i=0;i<weights.length;i++) {
			layersSize[i+1]=weights[i].length;
		}
	}
	public void prepareError() {
		error=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			error[i]=new float[weights[i].length];
		}
	}
	
	public void initializeOpenCL() {
		CL.setExceptionsEnabled(true);
		
		final int platformIndex=1;
		final int deviceIndex=0;
		
		int[] platformNumber=new int[1];
		CL.clGetPlatformIDs(1, null, platformNumber);
		cl_platform_id platforms[]=new cl_platform_id[platformNumber[0]];
		CL.clGetPlatformIDs(platformNumber[0], platforms, null);
		
		int[] deviceNumber=new int[1];
		CL.clGetDeviceIDs(platforms[platformIndex], CL.CL_DEVICE_TYPE_ALL, 1, null, deviceNumber);
		cl_device_id[] devices=new cl_device_id[deviceNumber[0]];
		CL.clGetDeviceIDs(platforms[platformIndex], CL.CL_DEVICE_TYPE_ALL, deviceNumber[0], devices, null);
		
		cl_context_properties contextProperties=new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platforms[platformIndex]);
		
		context=CL.clCreateContext(contextProperties, 1, new cl_device_id[] {devices[deviceIndex]}, null, null, null);
		
		cl_queue_properties queueProperties=new cl_queue_properties();
		//queueProperties.addProperty(CL.CL_QUEUE_PROFILING_ENABLE, 1);
		commandQueue=CL.clCreateCommandQueueWithProperties(context, devices[deviceIndex], queueProperties, null);
		
		String[] programs=new String[openCLProgram.size()+1];
		String[] temp=openCLProgram.values().toArray(new String[0]);
		for(int i=0;i<temp.length;i++) {
			programs[i+1]=temp[i];
		}
		programs[0]=function.getOpenCLProgram();
		program=CL.clCreateProgramWithSource(context, programs.length, programs, null, null);
		CL.clBuildProgram(program, 1, new cl_device_id[] {devices[deviceIndex]}, null, null, null);
		
		if(openCLProgram.containsKey("simulate"))			simulateKernel=CL.clCreateKernel(program, "simulate", null);
		if(openCLProgram.containsKey("outputError"))		outputLayerErrorKernel=CL.clCreateKernel(program, "outputError", null);
		if(openCLProgram.containsKey("calculateError"))		calculateErrorKernel=CL.clCreateKernel(program, "calculateError", null);
		if(openCLProgram.containsKey("calculateWeights"))	calculateWeightsKernel=CL.clCreateKernel(program, "calculateWeights", null);
		
		prepareCLMem();
		
		openCLLoaded=true;
	}
	protected void prepareCLMem() {
		weightsCL=new cl_mem[weights.length];
		deltaWeightsCL=new cl_mem[weights.length];
		outputsCL=new cl_mem[weights.length];
		errorCL=new cl_mem[weights.length];
		
		float[] wagiCLSrc;
		
		int index;
		for(int i=0;i<weights.length;i++) {
			//int connectionsNumber=;
			wagiCLSrc=new float[weights[i].length*(i==0?inputsNumber+1:weights[i-1].length+1)];
			
			index=0;
			for(int j=0;j<weights[i].length;j++) {
				for(int k=0;k<weights[i][j].length;k++) {
					wagiCLSrc[index]=weights[i][j][k];
					index++;
				}
			}
			weightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(index+1), Pointer.to(wagiCLSrc), null);
			deltaWeightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*(index+1), null, null);
			
			outputsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(layersSize[i+1]+1), Pointer.to(new float[] {1}), null);
			errorCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*(layersSize[i+1]), null, null);								//TODO NN add host_no_acess flag
			//System.out.println("layersSize["+(i+1)+"]="+layersSize[i+1]);
			//System.out.println("error["+i+"].length="+error[i].length);
		}
		if(learningSequence!=null)
			for(LearningSequence ls:learningSequence)
				ls.initializeCL(context,commandQueue);
	}
	protected void clearCLMem() {
		for(int i=0;i<weights.length;i++) {
			CL.clReleaseMemObject(weightsCL[i]);
			CL.clReleaseMemObject(deltaWeightsCL[i]);
			CL.clReleaseMemObject(outputsCL[i]);
			CL.clReleaseMemObject(errorCL[i]);
		}
		
		for(LearningSequence le:learningSequence) {
			le.clearCL();
		}
	}
	public void clearCPUData() {
		if(openCLLoaded) {
			weights=null;
			deltaWeights=null;
			error=null;
			output=null;
		}else {
			throw new NeuralException(0);
		}
	}
	public void clearOpenCL() {
		openCLLoaded=false;
		
		clearCLMem();
		
		if(simulateKernel!=null)CL.clReleaseKernel(simulateKernel);
		if(outputLayerErrorKernel!=null)CL.clReleaseKernel(outputLayerErrorKernel);
		if(calculateErrorKernel!=null)CL.clReleaseKernel(calculateErrorKernel);
		if(calculateWeightsKernel!=null)CL.clReleaseKernel(calculateWeightsKernel);
		if(program!=null)CL.clReleaseProgram(program);
		
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
	}
	
	public final void setWeights(float[][][] weights) {
		if(!learning) {
			this.weights=weights;
			inputsNumber=weights[0][0].length;
			
			prepareData();
			
			if(openCLLoaded) {
				clearCLMem();
				prepareCLMem();
			}
		}else
			throw new NeuralException(1);
	}
	public void setLS(LearningSequence[] ls) {
		if(!learning) {
			learningSequence=ls;
			
			if(openCLLoaded) {
				for(LearningSequence le:learningSequence) {
					le.initializeCL(context,commandQueue);
				}
			}
		}else
			throw new NeuralException(1);
	}
	public final void setFunction(Function function) {
		if(!learning) {
			this.function=function;
		}else
			throw new NeuralException(1);
	}
	public void setThreadsNumber(int threads) {
		threadsNumber=threads;
	}
	
	public final int getInputNumber() {
		return inputsNumber;
	}
	public final int getOutputNumber(){
		return layersSize[layersNumber];
	}
	public final int getLSLenght() {
		return learningSequence.length;
	}
	public final LearningSequence[] getLS() {
		return learningSequence;
	}
	public final float[] getOutput() {
		if(openCLLoaded) {
			if(output==null) {
				float[] outputs=new float[layersSize[layersNumber]];
				CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, 0, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(outputs), 0, null, null);
				return outputs;
			}
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, 0, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(output[layersNumber-1]), 0, null, null);
		}
		return output[layersNumber-1];
	}
	public final float[][][] getWeights(){
		return weights;
	}
	public final Function getFunction() {
		return function;
	}
	
	public int getThreadsNumber() {
		return threadsNumber;
	}
	
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	public final boolean isLearnning() {
		return learning;
	}
	
	public void mixLS(Random random){
		int ilEl=learningSequence.length;
		LearningSequence[] newLS=new LearningSequence[ilEl];
		boolean[] included=new boolean[ilEl];									//True if LS elemnent is already in newLS
		
		int index;
				
		for(LearningSequence cu:learningSequence){
			while(true){
				index=random.nextInt(ilEl);
				if(!included[index]){
					newLS[index]=cu;
					included[index]=true;
					break;
				}
			}
		}
		
		learningSequence=newLS;
	}
	
	public void startLearning() {
		if(!learning) {
			learning=true;
		}else throw new NeuralException(3);
	}
	public void lSimulate(int nrElement) {
		if(openCLLoaded) {
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				cl_mem inputDataCL;
				if(nrLayer==0){															//Input layer
					inputDataCL=learningSequence[nrElement].inputsCL;
				}else{																	//Input layer
					inputDataCL=outputsCL[nrLayer-1];
				}
				
				int neurons=layersSize[nrLayer+1];
				int connections=layersSize[nrLayer]+1;
				
				CL.clSetKernelArg(simulateKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 1, Sizeof.cl_mem, Pointer.to(inputDataCL));
				CL.clSetKernelArg(simulateKernel, 2, Sizeof.cl_mem, Pointer.to(outputsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, simulateKernel, 1, null ,new long[] {neurons}, new long[]{1}, 0, null, null);
				}
		}else {
			float[] inputData = learningSequence[nrElement].inputs;
			
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				lsimulate(inputData,0,weights[nrLayer].length,nrLayer);
				
//				int length;
//				int end=-1;
//				Thread[] threads=new Thread[threadsNumber];
//				for(int i=0;i<threadsNumber;i++) {
//					length=(weights[nrLayer].length-end-1)/(threadsNumber-i);
//					SimRunnable runnable=new SimRunnable();
//					runnable.nrLayer=nrLayer;
//					runnable.start=end+1;
//					runnable.inputData=inputData;
//					end+=length;
//					runnable.end=end+1;
//					
//					threads[i]=new Thread(runnable);
//					threads[i].start();
//				}
//				for(int i=0;i<threadsNumber;i++) {
//					try {
//						threads[i].join();
//					} catch (InterruptedException e) {
//						Thread.currentThread().interrupt();
//						System.out.println("Learning interupted");
//					}
//				}
			}
		}
	}
	protected void lsimulate(float[] inputData,int start,int end,int layer) {
		if(layer>0)inputData=output[layer-1];
		
		for(int i=start;i<end;i++) {
			output[layer][i]=weights[layer][i][0];
			for(int j=1;j<weights[layer][i].length;j++){
				output[layer][i]+=weights[layer][i][j]*inputData[j-1];
			}
			
			output[layer][i]=function.function(output[layer][i]);
		}
	}
	public abstract void countWeights(int elementNr,float n,float m);
	public void update(int cycleNumber) {}
	public void endLearning() {
		if(openCLLoaded) {
			float[] weightsBuffer;
			int neurons;
			int connections;
			int index;
			
			if(weights==null) {
				weights=new float[layersNumber][][];
				
				for(int i=0;i<layersNumber;i++) {
					weights[i]=new float[layersSize[i+1]][layersSize[i]];
				}
			}
			
			for(int i=0;i<layersNumber;i++) {
				neurons=layersSize[i+1];
				connections=layersSize[i]+1;
				index=0;
				
				weightsBuffer=new float[neurons*connections];
				CL.clEnqueueReadBuffer(commandQueue, weightsCL[i], CL.CL_TRUE, 0, Sizeof.cl_float*weightsBuffer.length, Pointer.to(weightsBuffer), 0, null, null);
				
				for(int j=0;j<neurons;j++) {
					for(int k=0;k<connections;k++) {
						weights[i][j][k]=weightsBuffer[index++];
					}
				}
			}
		}
		learning=false;
	}
	
	protected final HashMap<String,String> openCLProgram=new HashMap<String,String>();
	
//	private class SimRunnable implements Runnable{				//Some code of multithreading feature
//		int start;
//		int end;
//		int nrLayer;
//		float[] inputData;
//		public void run() {
//			lsimulate(inputData,start,end,nrLayer);
//		}
//	}
//	private class TaskMenager{
//		Thread[] threads;
//		BlockingQueue<Runnable> tasks;
//		BlockingQueue<Integer> resoults;
//		int tasksNumber=0;
//		public TaskMenager(int threadsNumber) {
//			tasks=new ArrayBlockingQueue<Runnable>(4);
//			resoults=new ArrayBlockingQueue<Integer>(4);
//			
//			threads=new Thread[threadsNumber];
//			for(int i=0;i<threadsNumber;i++) {
//				threads[i]=new Thread() {
//					public void run() {
//						try {
//							tasks.take().run();
//							resoults.put(0);
//						} catch (InterruptedException e) {
//							Thread.currentThread().interrupt();
//							e.printStackTrace();
//						}
//					}
//				};
//				threads[i].setDaemon(true);
//				threads[i].start();
//			}
//		}
//		public void add(Runnable runnable) {
//			try {
//				tasks.put(runnable);
//				tasksNumber++;
//			} catch (InterruptedException e) {
//				Thread.currentThread().interrupt();
//				e.printStackTrace();
//			}
//		}
//		public void join() {
//			while(tasksNumber!=0) {
//				try {
//					resoults.take();
//					tasksNumber--;
//				} catch (InterruptedException e) {
//					Thread.currentThread().interrupt();
//					e.printStackTrace();
//				}
//			}
//		}
//	}
}