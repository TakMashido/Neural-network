package liblaries.neuralNetwork.learning;

import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

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
import liblaries.neuralNetwork.functions.OutputFunction;

public abstract class LNetwork{
	protected float[][][] weights;												//input layer have  ID=0	[a][b][c] a->layer, b->neuron, c->connection
	protected float[][][] deltaWeights;											//							[a][b][c]
	protected float[][] error;													//							[a][b]
	protected float[][] outputs;												//							[a][b]
	protected int inputsNumber;
	
	protected LearningSequence[] learningSequence;
	protected OutputFunction outputFunction;
	
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
	
	protected boolean supportsMultiThreading=false;
	protected int threadsNumber=1;
	protected WorkersCoordinator threadsCoordinator;
	
	protected boolean learning=false;
	
	public LNetwork(){addOpenCLProgram();}
	public LNetwork(float[][][] weights, OutputFunction outputFunction) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.outputFunction=outputFunction;
		addOpenCLProgram();
	}
	public LNetwork(float[][][] weights, OutputFunction outputFunction, boolean initializeOpenCL) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.outputFunction=outputFunction;
		
		addOpenCLProgram();
		if(initializeOpenCL)
			initializeOpenCL();
	}
	public LNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction) {
		createLNetwork(inputsNumber, layersSize, outputFunction);
		prepareData();
		addOpenCLProgram();
	}
	public LNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction, boolean initializeOpenCL) {
		createLNetwork(inputsNumber, layersSize, outputFunction);
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
	protected void createLNetwork(int inputsNumber,int[] layersSize,OutputFunction outputFunction){
		this.inputsNumber=inputsNumber;
		
		layersNumber=layersSize.length;
		this.layersSize=new int[layersNumber+1];
		this.layersSize[0]=inputsNumber;
		System.arraycopy(layersSize, 0, this.layersSize, 1, layersNumber);
		
		weights=new float[layersSize.length][][];
		randomizeWeights();
		
		this.outputFunction=outputFunction;
	}
	protected void prepareData() {
		deltaWeights=new float[weights.length][][];
		outputs=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			deltaWeights[i]=new float[weights[i].length][];
			outputs[i]=new float[weights[i].length];
			
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
	public void randomizeWeights() {
		Random random=new Random();
		for(int i=0;i<layersSize.length-1;i++){
			int connectionsNumber=layersSize[i]+1;
			if(weights[i]==null)
				weights[i]=new float[layersSize[i+1]][connectionsNumber];
			
			float maxWeighth=1/(float)(layersSize[i]/20+1)+0.000000000000000001f;
			
			for(int j=0;j<layersSize[i+1];j++){
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
		programs[0]=outputFunction.getOpenCLProgram();
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
			outputs=null;
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
	public final void setFunction(OutputFunction outputFunction) {
		if(!learning) {
			this.outputFunction=outputFunction;
		}else
			throw new NeuralException(1);
	}
	public void setThreadsNumber(int threads) {
		if(supportsMultiThreading)
			threadsNumber=threads;
		else throw new NeuralException(NeuralException.notSupportMultiThreading);
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
			if(outputs==null) {
				float[] outputs=new float[layersSize[layersNumber]];
				CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, 0, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(outputs), 0, null, null);
				return outputs;
			}
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, 0, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(outputs[layersNumber-1]), 0, null, null);
		}
		return outputs[layersNumber-1];
	}
	public final float[][][] getWeights(){
		return weights;
	}
	public final OutputFunction getFunction() {
		return outputFunction;
	}
	public byte getSimMethodID() {
		return 0;
	}
	
	public int[] getLayersSize() {
		return layersSize;
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
	public final boolean isSupportOpenThreading() {
		return supportsMultiThreading;
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
		if(supportsMultiThreading) {
			threadsCoordinator=new WorkersCoordinator(threadsNumber);
		}
	}
	public void action(int elementNr,float n,float m) throws InterruptedException, BrokenBarrierException {
		if(supportsMultiThreading) {
			threadsCoordinator.action(elementNr,n,m);
		} else {
			simulate(elementNr);
			lCountWeights(elementNr,n,m);
			
		}
	}
	/**
	 * Base method to simulate. Invoke openCL simulation or one thread simulation.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 */
	public final void simulate(int elementNr) {
		if(openCLLoaded) {
			lSimulateOpenCL(elementNr);
		}else {
			lSimulate(elementNr);
		}
	}
	/**
	 * One thread simulation method.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 */
	protected void lSimulate(int elementNr) {
		for(int i=0;i<layersNumber;i++) {
			lSimulate(elementNr, 0, layersSize[i+1], i, 0);
		}
	}
	/**
	 * Method invoking openCL simuation. 
	 * @param elementNr index of input data in {@link #learningSequence}.
	 */
	protected void lSimulateOpenCL(int elementNr) {
		for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
			cl_mem inputDataCL;
			if(nrLayer==0){															//Input layer
				inputDataCL=learningSequence[elementNr].inputsCL;
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
	}
	/**
	 * Simulation method to use with multithreading.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 * @param start index of first simulated neuron.
	 * @param end index of last simulated neuron +1.
	 * @param layer simulated layer.
	 * @param threadID ID of invocing thread
	 */
	protected void lSimulate(int elementNr,int start,int end,int layer,int threadID){
		float[] inputData;
		if(layer>0)inputData=outputs[layer-1];
		else inputData=learningSequence[elementNr].inputs;
		
		for(int i=start;i<end;i++) {
			outputs[layer][i]=weights[layer][i][0];
			for(int j=1;j<weights[layer][i].length;j++){
				outputs[layer][i]+=weights[layer][i][j]*inputData[j-1];
			}
			
			outputs[layer][i]=outputFunction.function(outputs[layer][i]);
		}
	}
	
	/**
	 * Base method to update weights. Invoke openCL updating or one thread updating.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 * @param n Learning rate
	 * @param m Learning momentum
	 */
	public final void countWeights(int elementNr,float n,float m) {
		if(openCLLoaded) {
			lCountWeightsOpenCL(elementNr, n, m);
		} else {
			lCountWeights(elementNr, n, m);
		}
	}
	/**
	 * One thread method to update weighths.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 * @param n Learning rate
	 * @param m Learning momentum
	 */
	public void lCountWeights(int elementNr,float n,float m) {
		for(int i=0;i<layersNumber;i++)
			lCountWeights(elementNr, n, m, 0, layersSize[i+1], i, 0);
	}
	/**
	 * OpenCL method to update weights.
	 * @param elementNr index of input data in {@link #learningSequence}.
	 * @param n Learning rate
	 * @param m Learning momentum
	 */
	public void lCountWeightsOpenCL(int elementNr,float n,float m) {}
	/**
	 * Update weights method to use with multithreading
	 * @param elementNr Index of input data in {@link #learningSequence}.
	 * @param n Learning rate
	 * @param m Learning momentum
	 * @param start Index of first updated neuron.
	 * @param end Index of last updated neuron +1.
	 * @param layer Updated layer.
	 * @param threadID ID of invocing thread
	 */
	public void lCountWeights(int elementNr,float n,float m,int start,int end,int layer,int threaID) {}
	
	/**
	 * Method invoked on each learning cycle to update some network parameters.
	 * @param cycleNumber Actual learning cycle
	 */
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
		if(supportsMultiThreading)
			threadsCoordinator.stop();
		learning=false;
	}
	
	protected final HashMap<String,String> openCLProgram=new HashMap<String,String>();
	
	private class WorkersCoordinator{
		Worker[] workers;
		FastCyclicBarrier partBarrier;
		FastCyclicBarrier workBarrier;
		
		boolean stopping=false;
		
		int elementNr;
		float n;
		float m;
		
		public WorkersCoordinator(int workersNum) {
			partBarrier=new FastCyclicBarrier(workersNum);
			workBarrier=new FastCyclicBarrier(workersNum+1);
			workers=new Worker[workersNum];
			
			int[][] starts=new int[workersNum][layersNumber];
			int[][] ends=new int[workersNum][layersNumber];
			
			int length;
			int end=-1;
			for(int i=0;i<layersNumber;i++) {
				for(int j=0;j<threadsNumber;j++) {
					length=(layersSize[i+1]-end-1)/(threadsNumber-j);
					starts[j][i]=end+1;
					end+=length;
					ends[j][i]=end+1;
				}
			}
			
			for(int i=0;i<workers.length;i++) {
				workers[i]=new Worker(this,starts[i],ends[i],i);
				workers[i].setName("nnThread"+i);
				workers[i].setDaemon(true);
				workers[i].start();
			}
		}
		
		public void action(int elementNr,float n,float m) throws InterruptedException, BrokenBarrierException {
			this.elementNr=elementNr;
			this.n=n;
			this.m=m;
			workBarrier.await();
		}
		public void stop() {
			stopping=true;
			for(int i=0;i<workers.length;i++)
				workers[i].interrupt();
		}
	}
	private class Worker extends Thread{
		int threadID;
		
		int[] start;						//Neurons to proceder by this worker in each layer
		int[] end;
		WorkersCoordinator coordinator;
		
		public Worker(WorkersCoordinator coordinator,int[] start,int[] end,int threadID) {
			this.coordinator=coordinator;
			this.start=start;
			this.end=end;
			this.threadID=threadID;
		}
		
		public void run() {
			try {
				while(learning) {
					coordinator.workBarrier.await();
					
					for(int i=0;i<layersNumber;i++) {
						lSimulate(coordinator.elementNr,start[i],end[i],i,threadID);
						coordinator.partBarrier.await();
					}
					
					for(int i=layersNumber-1;i>-1;i--) {
						lCountWeights(coordinator.elementNr,coordinator.n,coordinator.m,start[i],end[i],i,threadID);
						coordinator.partBarrier.await();
					}
				}
			} catch (InterruptedException e) {
				if(!coordinator.stopping)
					e.printStackTrace();
				return;
			}
		}
	}
	private class FastCyclicBarrier{
		private int threadsNumber;						//-1
		private int waitingThreads=0;
		private Thread[] locked;
		private FastAtomicBoolean work=new FastAtomicBoolean(false);
		
		public FastCyclicBarrier(int threadsNumber) {
			this.threadsNumber=threadsNumber-1;
			locked=new Thread[this.threadsNumber];
		}
		
		public void await() throws InterruptedException {
			while(!work.compareAndSet(false, true));						//Wait loop
			
			synchronized(locked) {
				if(waitingThreads<threadsNumber) {
					locked[waitingThreads]=Thread.currentThread();
					waitingThreads++;
				} else {
					for(int i=0;i<locked.length;i++) {
						LockSupport.unpark(locked[i]);
						waitingThreads=0;
					}
					work.set(false);
					return;
				}
			}
			
			work.set(false);
			
			LockSupport.park();
			if(Thread.currentThread().isInterrupted())
				throw new InterruptedException();
		}
		
		public int getThreadsNumber() {
			return threadsNumber+1;
		}
	}
	private class FastAtomicBoolean {
		private volatile boolean value;
		
		public FastAtomicBoolean(boolean value) {
			this.value=value;
		}
		
		public boolean compareAndSet(boolean excepted, boolean update) {
			if(value==excepted) {
				excepted=update;
				return true;
			} else {
				return false;
			}
		}
		
		public void set(boolean value) {
			this.value=value;
		}
	}
	
//	private class SimRunnable implements Runnable{				//Some code of not ready multithreading feature
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