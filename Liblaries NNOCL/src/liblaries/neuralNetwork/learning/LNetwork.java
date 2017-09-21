package liblaries.neuralNetwork.learning;

import java.util.Arrays;
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

public class LNetwork{
	private float[][][] weights;												//input layer have  ID=0	[a][b][c] a->layer, b->neuron, c->connection
	private float[][][] deltaWeights;											//							[a][b][c]
	private float[][] error;													//							[a][b]
	private float[][] output;													//							[a][b]
	private int inputsNumber;
	
	private LearningSeqence[] learningSeqence;
	private Function function;
	
	int[] layersSize;															//[0] inputNumber, [1] neurons in layer 0, [2] neurons in layer 1, ...
	int layersNumber;
	
	private boolean openCLLoaded=false;
	private cl_context context;
	private cl_command_queue commandQueue;
	
	private cl_program program;
	private cl_kernel simulateKernel;
	private cl_kernel outputLayerErrorKernel;
	private cl_kernel calculateErrorKernel;
	private cl_kernel calculateWeightsKernel;
	
	private cl_mem[] weightsCL;							//[a] a->layer
	private cl_mem[] deltaWeightsCL;					//[a]
	private cl_mem[] outputsCL;							//[a]			//with bias
	private cl_mem[] errorCL;							//[a]			//without bias
	
	private boolean learning=false;
	
	public LNetwork(){}
	public LNetwork(float[][][] weights, Function function) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.function=function;
	}
	public LNetwork(float[][][] weights, Function function, boolean loadOpenCL) {
		this.weights=weights;
		inputsNumber=weights[0][0].length-1;
		prepareData();
		
		this.function=function;
		
		if(loadOpenCL)
			initializeOpenCL();
	}
	public LNetwork(int inputsNumber,int[] layersSize, Function function) {
		createLNetwork(inputsNumber, layersSize, function);
		prepareData();
	}
	public LNetwork(int inputsNumber,int[] layersSize, Function function, boolean initializeOpenCL) {
		createLNetwork(inputsNumber, layersSize, function);
		if(initializeOpenCL) {
			initializeOpenCL();
		}
		else
			prepareData();
	}
	private void createLNetwork(int inputsNumber,int[] layersSize,Function function){
		this.inputsNumber=inputsNumber;
		inputsNumber++;
		
		Random random=new Random();
		
		for(int i=0;i<layersSize.length;i++){
			if(i==0){
				weights[i]=new float[layersSize[i]][inputsNumber];
				
				float maxWeighth=1/(float)(layersSize[i]/20+1)+0.000000000000000001f;
				
				for(int j=0;i<layersSize[i];j++){					
					for(int k=0;i<inputsNumber;k++){
						while(true){
							float waga=random.nextFloat();
							
							if(waga!=0){
								weights[i][j][k]=waga%maxWeighth;
								break;
							}
						}
					}
				}
			}
			else{
				weights[i]=new float[layersSize[i]][weights[i-1].length];
				
				float maxWeight=1/(float)(layersSize[i]/20+1)+0.000000000000000001f;
				
				for(int j=0;i<layersSize[i];j++){					
					for(int k=0;i<layersSize[i-1];k++){
						weights[i][j][k]=random.nextFloat()%maxWeight;
					}
				}
			}
		}
		
		this.function=function;
	}
	private void prepareData() {
		deltaWeights=new float[weights.length][][];
		output=new float[weights.length][];
		error=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			deltaWeights[i]=new float[weights[i].length][];
			output[i]=new float[weights[i].length];
			error[i]=new float[weights[i].length];
			
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
	
	public void initializeOpenCL() {
		CL.setExceptionsEnabled(true);
		
		final int platformIndex=0;
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
		commandQueue=CL.clCreateCommandQueueWithProperties(context, devices[deviceIndex], queueProperties, null);
		
		program=CL.clCreateProgramWithSource(context, 2, new String[] {openCLprogram,function.getOpenCLProgram()}, null, null);
		CL.clBuildProgram(program, 1, new cl_device_id[] {devices[deviceIndex]}, null, null, null);
		
		simulateKernel=CL.clCreateKernel(program, "simulate", null);
		outputLayerErrorKernel=CL.clCreateKernel(program, "outputError", null);
		calculateErrorKernel=CL.clCreateKernel(program, "calculateError", null);
		calculateWeightsKernel=CL.clCreateKernel(program, "calculateWeights", null);
		
		prepareCLMem();
		
		openCLLoaded=true;
	}
	private void prepareCLMem() {
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
			errorCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_float*(layersSize[i+1]), null, null);								//TODO NN add host_no_acess flag
			//System.out.println("layersSize["+(i+1)+"]="+layersSize[i+1]);
			//System.out.println("error["+i+"].length="+error[i].length);
		}
		if(learningSeqence!=null)
			for(LearningSeqence ls:learningSeqence)
				ls.initializeCL(context,commandQueue);
	}
	private void clearCLMem() {
		for(int i=0;i<weights.length;i++) {
			CL.clReleaseMemObject(weightsCL[i]);
			CL.clReleaseMemObject(deltaWeightsCL[i]);
			CL.clReleaseMemObject(outputsCL[i]);
			CL.clReleaseMemObject(errorCL[i]);
		}
		
		for(LearningSeqence le:learningSeqence) {
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
		
		for(int i=0;i<layersNumber;i++) {
			CL.clReleaseMemObject(weightsCL[i]);
			CL.clReleaseMemObject(deltaWeightsCL[i]);
			CL.clReleaseMemObject(outputsCL[i]);
			CL.clReleaseMemObject(errorCL[i]);
		}
		
		CL.clReleaseKernel(simulateKernel);
		CL.clReleaseKernel(outputLayerErrorKernel);
		CL.clReleaseKernel(calculateErrorKernel);
		CL.clReleaseKernel(calculateWeightsKernel);
		CL.clReleaseProgram(program);
		
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
	public final void setLS(LearningSeqence[] ls) {
		if(!learning) {
			learningSeqence=ls;
			
			if(openCLLoaded) {
				for(LearningSeqence le:learningSeqence) {
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
	
	public final int getInputNumber() {
		return inputsNumber;
	}
	public int getOutputNumber(){
		return layersSize[layersNumber];
	}
	public final int getLSLenght() {
		return learningSeqence.length;
	}
	public final LearningSeqence[] getCU() {
		return learningSeqence;
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
	
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	public final boolean isLearnning() {
		return learning;
	}
	
	public void startLearning() {
		if(!learning) {
			learning=true;
		}else throw new NeuralException(3);
	}
	public void lSimulate(int nrElement){
		if(openCLLoaded) {
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				cl_mem inputDataCL;
				if(nrLayer==0){															//Input layer
					inputDataCL=learningSeqence[nrElement].inputsCL;
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
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				float[] inputData;
				if(nrLayer==0){															//Input layer
					inputData=learningSeqence[nrElement].inputs;
				}
				else{																	//Output layer
					inputData=output[nrLayer-1];
				}
				
				for(int i=0;i<weights[nrLayer].length;i++) {
					output[nrLayer][i]=weights[nrLayer][i][0];
					for(int j=1;j<weights[nrLayer][i].length;j++){
						output[nrLayer][i]+=weights[nrLayer][i][j]*inputData[j-1];
					}
					
					output[nrLayer][i]=function.function(output[nrLayer][i]);
				}
			}
		}
	}
	public void coutError(int nrElementu){
		boolean pom=true;
		
		if(openCLLoaded) {
			CL.clSetKernelArg(outputLayerErrorKernel, 0, Sizeof.cl_mem, Pointer.to(errorCL[layersNumber-1]));			//Input layer
			CL.clSetKernelArg(outputLayerErrorKernel, 1, Sizeof.cl_mem, Pointer.to(outputsCL[layersNumber-1]));
			CL.clSetKernelArg(outputLayerErrorKernel, 2, Sizeof.cl_mem, Pointer.to(learningSeqence[nrElementu].outputsCL));
			CL.clEnqueueNDRangeKernel(commandQueue, outputLayerErrorKernel, 1, null, new long[] {layersSize[layersNumber-1]}, new long[] {1}, 0, null, null);
			
			for(int nrLayer=layersNumber-2;nrLayer>-1;nrLayer--){													//Hidden layer
				int neurons=layersSize[nrLayer+1];			//WTF? this work
				int connections=layersSize[nrLayer+1];
				
				CL.clSetKernelArg(calculateErrorKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer+1]));
				CL.clSetKernelArg(calculateErrorKernel, 1, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer+1]));
				CL.clSetKernelArg(calculateErrorKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer]));
				CL.clSetKernelArg(calculateErrorKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections+1}));
				CL.clEnqueueNDRangeKernel(commandQueue, calculateErrorKernel, 1, null, new long[] {neurons}, new long[] {1}, 0, null, null);
			}
		}else {
			for(int nrLayer=weights.length-1;nrLayer>-1;nrLayer--){
				if(pom){																							//Input layer
					for(int i=0;i<weights[nrLayer].length;i++){
						error[nrLayer][i]=learningSeqence[nrElementu].outputs[i]-output[nrLayer][i];				//Dont't have to reset error
					}
					pom=false;
				}else{																								//Hidden layer
					for(int i=0;i<weights[nrLayer].length;i++)														//Reset error
						error[nrLayer][i]=0;
					
					for(int i=0;i<weights[nrLayer+1].length;i++){
						for(int j=0;j<weights[nrLayer+1][i].length-1;j++){
							error[nrLayer][j]+=error[nrLayer+1][i]*weights[nrLayer+1][i][j+1];
						}
					}
				}
			}
		}
	}
	public void countWeights(int NrElementu,float n,float m){
		if(openCLLoaded) {														//Debug messages(for me)
			float[][] warstwa =new float[2][];
			warstwa[0]=new float[weights[0].length*(inputsNumber+1)];
			warstwa[1]=new float[weights[1].length*weights[1][0].length];
			CL.clEnqueueReadBuffer(commandQueue, weightsCL[0], CL.CL_TRUE, 0, Sizeof.cl_float*warstwa[0].length, Pointer.to(warstwa[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, weightsCL[1], CL.CL_TRUE, 0, Sizeof.cl_float*warstwa[1].length, Pointer.to(warstwa[1]), 0, null, null);
			System.out.println("wagi=    "+Arrays.toString(warstwa[0])+" "+Arrays.toString(warstwa[1]));
			
			//System.out.println("error[0].length"+error[0].length);
			//System.out.println("error[1].length"+error[1].length);
			CL.clEnqueueReadBuffer(commandQueue, errorCL[0], CL.CL_TRUE, 0, Sizeof.cl_float*error[0].length, Pointer.to(error[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, errorCL[1], CL.CL_TRUE, 0, Sizeof.cl_float*error[1].length, Pointer.to(error[1]), 0, null, null);
			System.out.println("b³¹d=    "+Arrays.toString(error[0])+", "+Arrays.toString(error[1]));
			
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[0], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*output[0].length, Pointer.to(output[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[1], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*output[1].length, Pointer.to(output[1]), 0, null, null);
			System.out.println("wyjœcia= "+Arrays.toString(output[0])+" "+Arrays.toString(output[1]));
		}else {
			System.out.println("wagi=    "+Arrays.toString(weights[0][0])+" "+Arrays.toString(weights[0][1])+" "+Arrays.toString(weights[0][2])+", "+Arrays.toString(weights[1][0])+" "+Arrays.toString(weights[1][1]));
			System.out.println("b³¹d=    "+Arrays.toString(error[0])+", "+Arrays.toString(error[1]));
			System.out.println("wyjœcia= "+Arrays.toString(output[0])+" "+Arrays.toString(output[1]));
		}
		System.out.println();
		
		if(openCLLoaded) {
			//long time=System.nanoTime();
			CL.clSetKernelArg(calculateWeightsKernel, 5, Sizeof.cl_float, Pointer.to(new float[] {n}));
			CL.clSetKernelArg(calculateWeightsKernel, 6, Sizeof.cl_float, Pointer.to(new float[] {m}));
			for(int i=layersNumber-1;i>-1;i--) {
				long time=System.nanoTime();
				CL.clSetKernelArg(calculateWeightsKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 1, Sizeof.cl_mem, Pointer.to(deltaWeightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 3, Sizeof.cl_mem, Pointer.to(i==0?learningSeqence[i].inputsCL:outputsCL[i-1]));
				CL.clSetKernelArg(calculateWeightsKernel, 4, Sizeof.cl_int, Pointer.to(new int[] {layersSize[i]+1}));
				System.out.println("settingup time="+((System.nanoTime()-time)/1000)/1000f+"ms");
				time=System.nanoTime();
				CL.clEnqueueNDRangeKernel(commandQueue, calculateWeightsKernel, 2, null, new long[] {layersSize[i+1],layersSize[i]+1}, new long[] {1,1}, 0, null, null);
				System.out.println("invoking time="+((System.nanoTime()-time)/1000)/1000f+"ms");
			}
			//System.out.println("time="+((System.nanoTime()-time)/1000)/1000f+"ms");
		}else {
			//long time=System.nanoTime();
			for(int i=0;i<weights.length;i++){
				for(int j=0;j<weights[i].length;j++){
					for(int k=0;k<weights[i][j].length;k++){
						float delta=m*deltaWeights[i][j][k]+n*error[i][j]*(k==0?1:(i==0?learningSeqence[NrElementu].inputs[k-1]:output[i-1][k-1]));					//FIXME NN find where put derivative of function
						
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
				}
			}
			//System.out.println("time="+((System.nanoTime()-time)/1000)/1000f+"ms");
		}
	}
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
				connections=layersSize[i];
				index=0;
				
				weightsBuffer=new float[layersSize[i+1]*layersSize[i]];
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
	public void mixLS(Random random){
		int ilEl=learningSeqence.length;
		LearningSeqence[] newLS=new LearningSeqence[ilEl];
		boolean[] included=new boolean[ilEl];									//True if LS elemnent is already in newLS
		
		int index;
				
		for(LearningSeqence cu:learningSeqence){
			while(true){
				index=random.nextInt(ilEl);
				if(!included[index]){
					newLS[index]=cu;
					included[index]=true;
					break;
				}
			}
		}
		
		learningSeqence=newLS;
	}
	
	private static final String openCLprogram=
			  "__kernel void outputError(__global float *error, __global float *outputs,__global float *goodOutputs){"
			+ "		int neuron=get_global_id(0);"
			+ "		error[neuron]=goodOutputs[neuron]-outputs[neuron+1];"
			+ "}\n"
			+ ""
			+ "__kernel void calculateError(__global const float *weights, __global const float *errorUp, __global float *error, int connectionsNumber){"
			+ "		int neuron=get_global_id(0);"
			+ "		int neuron1=neuron+1;"
			+ "		"
			+ "		error[neuron]=0;"
			+ "		for(int i=0;i<connectionsNumber;i++){"
			+ "			error[neuron]+=errorUp[i]*weights[neuron1+i*connectionsNumber];"
			+ "		}"
			+ "}"
			+ ""
			+ "__kernel void calculateWeights(__global float *weights,__global float *deltaWeights,__global float *error,__global float *input,int connectionsNumber, float n, float m){"
			+ "		int neuron=get_global_id(0);"
			+ "		int connection=get_global_id(1);"
			+ "		int index=neuron*connectionsNumber+connection;"
			+ "		"
			+ "		float delta=n*input[connection]*error[neuron]+m*deltaWeights[index];"						//FIXME SNOCL find where put the derivative of a function
			+ "		weights[index]+=delta;"
			+ "		deltaWeights[index]=delta;"
			+ "}";
}