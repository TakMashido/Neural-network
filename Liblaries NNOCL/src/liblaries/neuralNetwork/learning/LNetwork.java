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
	private float[][] outputs;													//							[a][b]
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
	private cl_mem[] outputsCL;							//[a]
	private cl_mem[] errorCL;							//[a]
	
	private boolean learning=false;
	
	public LNetwork(){}
	public LNetwork(int inputsNumber,int[] layersSize,Function function) {
		createLNetwork(inputsNumber, layersSize, function);
		prepareData();
	}
	public LNetwork(int inputsNumber,int[] layersSize,Function function,boolean openCLUse) {
		createLNetwork(inputsNumber, layersSize, function);
		if(openCLUse) {
			initializeOpenCL();
		}
		else
			prepareData();
	}
	private void createLNetwork(int inputsNumber,int[] layersSize,Function function){
		this.inputsNumber=inputsNumber;
		
		Random random=new Random();
		
		for(int i=0;i<layersSize.length;i++){
			if(i==0){
				weights[i]=new float[layersSize[i]][inputsNumber];
				
				float maxWaga=1/(float)(layersSize[i]/20+1)+0.000000000000000001f;
				
				for(int j=0;i<layersSize[i];j++){					
					for(int k=0;i<inputsNumber;k++){
						while(true){
							float waga=random.nextFloat();
							
							if(waga!=0){
								weights[i][j][k]=waga%maxWaga;
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
	private void prepareCLMem() {
		weightsCL=new cl_mem[weights.length];
		deltaWeightsCL=new cl_mem[weights.length];
		outputsCL=new cl_mem[weights.length];
		errorCL=new cl_mem[weights.length];
		
		float[] wagiCLSrc;
		
		int index;
		for(int i=0;i<weights.length;i++) {
			int liczbaPo³¹czeñ=i==0?inputsNumber:weights[i-1].length;
			wagiCLSrc=new float[weights[i].length*liczbaPo³¹czeñ];
			
			index=0;
			for(int j=0;j<weights[i].length;j++) {
				for(int k=0;k<weights[i][j].length;k++) {
					wagiCLSrc[index]=weights[i][j][k];
					index++;
				}
			}
			weightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(index+1), Pointer.to(wagiCLSrc), null);
			deltaWeightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*(index+1), null, null);
			
			outputsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_float*weights[i].length, null, null);
			errorCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_float*weights[i].length, null, null);								//TODO SNCL add host_no_acess flag
		}
		for(LearningSeqence le:learningSeqence) {
			le.initializeCL(context);
		}
	}
	public void prepareData() {
		deltaWeights=new float[weights.length][][];
		outputs=new float[weights.length][];
		error=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			deltaWeights[i]=new float[weights[i].length][];
			outputs[i]=new float[weights[i].length];
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
			prepareData();
			
			if(openCLLoaded) {
				clearCLMem();
				prepareCLMem();
			}
		}else
			throw new NeuralException(1);
	}
	public final void setInputNumber(int inputNumber) {
		if(!learning)
			this.inputsNumber=inputNumber;
		else
			throw new NeuralException(1);
	}
	public final void setLS(LearningSeqence[] ls) {
		if(!learning) {
			learningSeqence=ls;
			
			if(openCLLoaded) {
				for(LearningSeqence le:learningSeqence) {
					le.initializeCL(context);
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
	public final int getLSLenght() {
		return learningSeqence.length;
	}
	public final LearningSeqence[] getCU() {
		return learningSeqence;
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
	public final Function getFunction() {
		return function;
	}
	
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	
	public void startLearning() {
		if(!learning) {
			learning=true;
		}else throw new NeuralException(3);
	}
	public void LSimulateNetwork(int nrElement){
		if(openCLLoaded) {
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				cl_mem inputDataCL;
				if(nrLayer==0){															//Input layer
					inputDataCL=learningSeqence[nrElement].outputsCL;
				}else{																	//Input layer
					inputDataCL=outputsCL[nrLayer-1];
				}
				
				int neurons=layersSize[nrLayer+1];
				int connections=layersSize[nrLayer];
				
				/*cl_mem preOutputCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*neurons*connections, null, null);
				
				CL.clSetKernelArg(simulateKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 1, Sizeof.cl_mem, Pointer.to(inputDataCL));
				CL.clSetKernelArg(simulateKernel, 2, Sizeof.cl_mem, Pointer.to(preOutputCL));
				CL.clSetKernelArg(simulateKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, simulateKernel, 2, null ,new long[] {neurons,connections}, new long[]{1,1}, 0, null, null);
								
				CL.clSetKernelArg(countOutputsKernel, 0, Sizeof.cl_mem, Pointer.to(preOutputCL));
				CL.clSetKernelArg(countOutputsKernel, 1, Sizeof.cl_mem, Pointer.to(outputsCL[nrLayer]));
				CL.clSetKernelArg(countOutputsKernel, 2, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, countOutputsKernel, 1, null, new long[] {neurons}, new long[] {1}, 0, null, null);
				
				CL.clReleaseMemObject(preOutputCL);*/
				
				CL.clSetKernelArg(simulateKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 1, Sizeof.cl_mem, Pointer.to(inputDataCL));
				CL.clSetKernelArg(simulateKernel, 2, Sizeof.cl_mem, Pointer.to(outputsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, simulateKernel, 1, null ,new long[] {neurons}, new long[]{1,1}, 0, null, null);
				
				//CL.clEnqueueReadBuffer(commandQueue, outputsCL[nrLayer], CL.CL_TRUE, 0, Sizeof.cl_float*outputs[nrLayer].length, Pointer.to(outputs[nrLayer]), 0, null, null);
				}
		}else {
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				float[] inputData;
				if(nrLayer==0){														//Warstwa wejœciowa
					inputData=learningSeqence[nrElement].inputs;
				}
				else{																	//Warstwa wyjœciowa
					inputData=outputs[nrLayer-1];
				}
				
				for(int i=0;i<weights[nrLayer].length;i++) {
					outputs[nrLayer][i]=0;
					for(int j=0;j<weights[nrLayer][i].length;j++){
						outputs[nrLayer][i]+=weights[nrLayer][i][j]*inputData[j];
					}
					
					outputs[nrLayer][i]=function.function(outputs[nrLayer][i]);
				}
			}
		}
	}
	public void coutError(int nrElementu){
		boolean pom=true;																			//Zmienna pomocnicza, optymalizuj¹ca
		
		if(openCLLoaded) {
			for(int nrLayer=layersNumber-1;nrLayer>-1;nrLayer--){
				if(pom){
					CL.clSetKernelArg(outputLayerErrorKernel, 0, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer]));
					CL.clSetKernelArg(outputLayerErrorKernel, 1, Sizeof.cl_mem, Pointer.to(outputsCL[nrLayer]));
					CL.clSetKernelArg(outputLayerErrorKernel, 2, Sizeof.cl_mem, Pointer.to(learningSeqence[nrElementu].outputsCL));
					CL.clEnqueueNDRangeKernel(commandQueue, outputLayerErrorKernel, 1, null, new long[] {layersSize[layersNumber]}, new long[] {1,1}, 0, null, null);
					pom=false;
				}else{																					//warstwy ukryte
					int neurons=layersSize[nrLayer+1];
					int connections=layersSize[nrLayer+1];
					
					CL.clSetKernelArg(calculateErrorKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer+1]));
					CL.clSetKernelArg(calculateErrorKernel, 1, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer+1]));
					CL.clSetKernelArg(calculateErrorKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer]));
					CL.clSetKernelArg(calculateErrorKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
					CL.clEnqueueNDRangeKernel(commandQueue, calculateErrorKernel, 1, null, new long[] {neurons}, new long[] {1,1}, 0, null, null);
				}
			}
		}else {
			for(int nrLayer=weights.length-1;nrLayer>-1;nrLayer--){
				if(pom){																				//warstwa wyjœciowa
					for(int i=0;i<weights[nrLayer].length;i++){
						error[nrLayer][i]=learningSeqence[nrElementu].outputs[i]-outputs[nrLayer][i];		//Nie ma potrzeby zerowania b³êdu
					}
					pom=false;
				}else{																					//warstwy ukryte
					for(int i=weights[nrLayer].length-1;i>-1;i--){										//Zerowanie b³êdu
						error[nrLayer][i]=0;
					}
					
					for(int i=weights[nrLayer+1].length-1;i>-1;i--){
						for(int j=weights[nrLayer+1][i].length-1;j>-1;j--){
							error[nrLayer][j]+=error[nrLayer+1][i]*weights[nrLayer+1][i][j];
						}
					}
				}
			}
			//for(NNeuron neuron:warstwa.neurony){												//XXX SNOCL eksperyment-Podstawienie b³¹du do pochodnej funkcji
			//	neuron.b³¹d*=funkcja.pochodna(neuron.b³¹d);
			//	neuron.b³¹d*=funkcja.pochodna(neuron.wyjœcie);
			//	neuron.b³¹d=funkcja.pochodna(neuron.b³¹d);
			//}
		}
	}
	public void countWeights(int NrElementu,float n,float m){
		if(openCLLoaded) {																																	//TODO NNOCL delete debug messages
			float[][] warstwa =new float[2][];
			warstwa[0]=new float[weights[0].length*inputsNumber];
			warstwa[1]=new float[weights[0].length*weights[1].length];
			CL.clEnqueueReadBuffer(commandQueue, weightsCL[0], CL.CL_TRUE, 0, Sizeof.cl_float*weights[0].length*inputsNumber, Pointer.to(warstwa[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, weightsCL[1], CL.CL_TRUE, 0, Sizeof.cl_float*weights[0].length*weights[1].length, Pointer.to(warstwa[1]), 0, null, null);
			System.out.println("wagi=    "+Arrays.toString(warstwa[0])+" "+Arrays.toString(warstwa[1]));
			
			CL.clEnqueueReadBuffer(commandQueue, errorCL[0], CL.CL_TRUE, 0, Sizeof.cl_float*error[0].length, Pointer.to(error[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, errorCL[1], CL.CL_TRUE, 0, Sizeof.cl_float*error[1].length, Pointer.to(error[1]), 0, null, null);
			System.out.println("b³¹d=    "+Arrays.toString(error[0])+", "+Arrays.toString(error[1]));
			
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[0], CL.CL_TRUE, 0, Sizeof.cl_float*outputs[0].length, Pointer.to(outputs[0]), 0, null, null);
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[1], CL.CL_TRUE, 0, Sizeof.cl_float*outputs[1].length, Pointer.to(outputs[1]), 0, null, null);
			System.out.println("wyjœcia= "+Arrays.toString(outputs[0])+" "+Arrays.toString(outputs[1]));
		}else {
			System.out.println("wagi=    "+Arrays.toString(weights[0][0])+" "+Arrays.toString(weights[0][1])+" "+Arrays.toString(weights[0][2])+" "+Arrays.toString(weights[1][0])+" "+Arrays.toString(weights[1][1]));
			System.out.println("b³¹d=    "+Arrays.toString(error[0])+", "+Arrays.toString(error[1]));
			System.out.println("wyjœcia= "+Arrays.toString(outputs[0])+" "+Arrays.toString(outputs[1]));
		}
		System.out.println();
		
		if(openCLLoaded) {
			CL.clSetKernelArg(calculateWeightsKernel, 5, Sizeof.cl_float, Pointer.to(new float[] {n}));
			CL.clSetKernelArg(calculateWeightsKernel, 6, Sizeof.cl_float, Pointer.to(new float[] {m}));
			
			for(int i=layersNumber-1;i>-1;i--) {
				CL.clSetKernelArg(calculateWeightsKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 1, Sizeof.cl_mem, Pointer.to(deltaWeightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 3, Sizeof.cl_mem, Pointer.to(i==0?learningSeqence[i].inputsCL:outputsCL[i-1]));
				CL.clSetKernelArg(calculateWeightsKernel, 4, Sizeof.cl_int, Pointer.to(new int[] {layersSize[i]}));
				CL.clEnqueueNDRangeKernel(commandQueue, calculateWeightsKernel, 2, null, new long[] {layersSize[i+1],layersSize[i]}, new long[] {1,1}, 0, null, null);
			}
		}else {			
			for(int i=0;i<weights.length;i++){
				for(int j=weights[i].length-1;j>-1;j--){
					for(int k=weights[i][j].length-1;k>-1;k--){
						//float wartoœæ=i==0?ci¹gUcz¹cy[NrElementu].wejœcia[k]:wyjœcia[i-1][k];				//Podane linijki zosta³y skrócone do nastêpnej
						//float delta=n*wartoœæ*b³¹d[i][j]+m*deltaWagi[i][j][k];
						float delta=m*deltaWeights[i][j][k]+n*error[i][j]*(i==0?learningSeqence[NrElementu].inputs[k]:outputs[i-1][k]);					//FIXME SNOCL okreœliæ miejsce zastosowania pochodnej funkcji we wzorze
						
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
				}
			}
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
	public int getOutputNumber(){
		return layersSize[layersNumber];
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
			+ "		error[neuron]=goodOutputs[neuron]-outputs[neuron];"
			+ "}\n"
			+ ""
			+ "__kernel void calculateError(__global const float *weights, __global const float *errorUp, __global float *error, int connectionsNumber){"
			+ "		int neuron=get_global_id(0);"
			+ "		"
			+ "		error[neuron]=0;"
			+ "		for(int i=0;i<connectionsNumber;i++){"
			+ "			error[neuron]+=errorUp[i]*weights[neuron+i*connectionsNumber];"
			+ "		}"
			+ "}"
			+ ""
			+ "__kernel void calculateWeights(__global float *weights,__global float *deltaWeights,__global float *error,__global float *input,int connectionsNumber, float n, float m){"
			+ "		int neuron=get_global_id(0);"
			+ "		int connection=get_global_id(1);"
			+ "		int index=neuron*connectionsNumber+connection;"
			+ "		"
			+ "		float delta=n*input[connection]*error[neuron]+m*deltaWeights[index];"						//FIXME SNOCL fint where put the derivative of a function
			+ "		weights[index]+=delta;"
			+ "		deltaWeights[index]=delta;"
			+ "}";
}