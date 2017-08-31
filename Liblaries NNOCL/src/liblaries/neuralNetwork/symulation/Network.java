package liblaries.neuralNetwork.symulation;

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

public class Network{
	public static final String wersjaBibliotek="1.3.1";
	private float[][][] weights;				//Warstwa wej�ciowa ma ID=0			[a][b][c] a->warstwa, b->neuron, c->po��czenie
	private float[][] wyj�cia;
	
	private int inputNumber;							//liczba informacji wej�ciowych do sieci
	
	public Function function;
	
	int[] layersSize;
	
	private boolean openCLLoaded=false;
	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private cl_kernel symulujKernel;
	private cl_kernel sumujKernel;
	
	private cl_mem[] wagiCL;							//[a] a->warstwa
	private cl_mem[] wyj�ciaCL;							//[a]
	
	//public Network() {}			//to be or not to be?
	public Network(int inputNumber, float[][][] weights, Function function) {
		setWeights(inputNumber,weights);
		this.function=function;
	}
	public Network(int inputNumber, float[][][] weights, Function function, boolean loadOpenCL) {
		setWeights(inputNumber,weights);
		this.function=function;
		
		if(loadOpenCL)
			initializeOpenCL();
	}
	public void setWeights(int inputNumber,float[][][] weights) {
		this.inputNumber=inputNumber;
		this.weights=weights;
		
		if(openCLLoaded)
			przygotujCLMem();
		
		inicjujWyj�cia();
		
		layersSize=new int[weights.length];
		for(int i=0;i<weights.length;i++) {
			layersSize[i]=weights[i].length;
		}
	}
	
	public void initializeOpenCL() {
		CL.setExceptionsEnabled(true);
		
		int[] platformNumber=new int[1];
		CL.clGetPlatformIDs(1, null, platformNumber);
		cl_platform_id platforms[]=new cl_platform_id[platformNumber[0]];
		CL.clGetPlatformIDs(platformNumber[0], platforms, null);
		
		int[] deviceNumber=new int[1];
		CL.clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_ALL, 1, null, deviceNumber);
		cl_device_id[] devices=new cl_device_id[deviceNumber[0]];
		CL.clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_ALL, deviceNumber[0], devices, null);
		
		cl_context_properties contextProperties=new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platforms[0]);
		
		context=CL.clCreateContext(contextProperties, 1, new cl_device_id[] {devices[0]}, null, null, null);
		
		cl_queue_properties queueProperties=new cl_queue_properties();
		commandQueue=CL.clCreateCommandQueueWithProperties(context, devices[0], queueProperties, null);
		
		program=CL.clCreateProgramWithSource(context, 2, new String[] {openCLprogram,function.getOpenCLProgram()}, null, null);
		CL.clBuildProgram(program, 1, new cl_device_id[] {devices[0]}, null, null, null);
		
		symulujKernel=CL.clCreateKernel(program, "symuluj", null);
		sumujKernel=CL.clCreateKernel(program, "sumujWyjscia", null);
		
		przygotujCLMem();
		
		openCLLoaded=true;
	}
	public void clearOpenCL() {
		openCLLoaded=false;
		
		for(int i=0;i<weights.length;i++) {
			CL.clReleaseMemObject(wagiCL[i]);
			CL.clReleaseMemObject(wyj�ciaCL[i]);
		}
		
		CL.clReleaseKernel(sumujKernel);
		CL.clReleaseKernel(symulujKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
	}
	private void przygotujCLMem() {
		wagiCL=new cl_mem[weights.length];
		wyj�ciaCL=new cl_mem[weights.length];
		
		float[] wagiCLSrc;
		
		int index;
		for(int i=0;i<weights.length;i++) {
			int liczbaPo��cze�=i==0?inputNumber:weights[i-1].length;
			wagiCLSrc=new float[weights[i].length*liczbaPo��cze�];
			
			index=0;
			for(int j=0;j<weights[i].length;j++) {
				for(int k=0;k<weights[i][j].length;k++) {
					wagiCLSrc[index]=weights[i][j][k];
					index++;
				}
			}
			wagiCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(index+1), Pointer.to(wagiCLSrc), null);
			
			wyj�ciaCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_float*wyj�cia[i].length, null, null);
		}
	}
	private void inicjujWyj�cia() {
		wyj�cia=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			wyj�cia[i]=new float[weights[i].length];
		}
	}
	public void clearCPUData() {
		if(openCLLoaded)
			weights=null;
		else throw new NeuralException(0);
	}
	
	public final int getInputNumber() {
		return inputNumber;
	}
	public final float[][][] getWeight() {
		return weights;
	}
	public final float[][] getOutputs() {
		return wyj�cia;
	}
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	
	
	public float[] symulujSie�(float[] daneWej�ciowe) {
		boolean pom=true;															//Zmienna optymalizuj�ca
		
		if(daneWej�ciowe.length!=inputNumber)
			throw new Error("Invalid input lenght. you use : "+daneWej�ciowe.length+" network lenght size: "+inputNumber);
		if(openCLLoaded) {
			for(int nrWarstwy=0;nrWarstwy<layersSize.length;nrWarstwy++){
				if(pom){																//Warstwa wej�ciowa
					pom=false;															//FIXME NN fix simulation
				}	
				else{																	//Warstwa wyj�ciowa
					daneWej�ciowe=wyj�cia[nrWarstwy-1];
				}
				int neurony=layersSize[nrWarstwy];
				int po��czenia=nrWarstwy==0?inputNumber:layersSize[nrWarstwy-1];
				
				cl_mem daneWej�cioweCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*daneWej�ciowe.length, Pointer.to(daneWej�ciowe), null);
				cl_mem preWyj�ciaCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*neurony*po��czenia, null, null);
				Pointer preWyj�ciaCLPtr=Pointer.to(preWyj�ciaCL);
				
				CL.clSetKernelArg(symulujKernel, 0, Sizeof.cl_mem, Pointer.to(wagiCL[nrWarstwy]));
				CL.clSetKernelArg(symulujKernel, 1, Sizeof.cl_mem, Pointer.to(daneWej�cioweCL));
				CL.clSetKernelArg(symulujKernel, 2, Sizeof.cl_mem, preWyj�ciaCLPtr);
				CL.clSetKernelArg(symulujKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {po��czenia}));
				CL.clEnqueueNDRangeKernel(commandQueue, symulujKernel, 2, null ,new long[] {neurony,po��czenia}, new long[]{1,1}, 0, null, null);
				
				CL.clSetKernelArg(sumujKernel, 0, Sizeof.cl_mem, preWyj�ciaCLPtr);
				CL.clSetKernelArg(sumujKernel, 1, Sizeof.cl_mem, Pointer.to(wyj�ciaCL[nrWarstwy]));
				CL.clSetKernelArg(sumujKernel, 2, Sizeof.cl_int, Pointer.to(new int[] {po��czenia}));
				CL.clEnqueueNDRangeKernel(commandQueue, sumujKernel, 1, null, new long[] {neurony}, new long[] {1}, 0, null, null);
				
				CL.clEnqueueReadBuffer(commandQueue, wyj�ciaCL[nrWarstwy], CL.CL_TRUE, 0, Sizeof.cl_float*neurony, Pointer.to(wyj�cia[nrWarstwy]), 0, null, null);
				
				CL.clReleaseMemObject(daneWej�cioweCL);
				CL.clReleaseMemObject(preWyj�ciaCL);
				
				for(int i=0;i<neurony;i++)
					wyj�cia[nrWarstwy][i]=function.function(wyj�cia[nrWarstwy][i]);
				
				CL.clReleaseMemObject(wyj�ciaCL[nrWarstwy]);
				wyj�ciaCL[nrWarstwy]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*wyj�cia[nrWarstwy].length, Pointer.to(wyj�cia[nrWarstwy]), null);
				}
		}else {
			for(int nrWarstwy=0;nrWarstwy<layersSize.length;nrWarstwy++){
				if(pom){																//Warstwa wej�ciowa
					pom=false;
				}
				else{																	//Warstwa wyj�ciowa
					daneWej�ciowe=wyj�cia[nrWarstwy-1];
				}
				
				for(int i=0;i<weights[nrWarstwy].length;i++) {
					wyj�cia[nrWarstwy][i]=0;
					for(int j=0;j<weights[nrWarstwy][i].length;j++){
						wyj�cia[nrWarstwy][i]+=weights[nrWarstwy][i][j]*daneWej�ciowe[j];
					}
					
					wyj�cia[nrWarstwy][i]=function.function(wyj�cia[nrWarstwy][i]);
				}
			}
		}
		return wyj�cia[layersSize.length-1];
	}
	
	private static final String openCLprogram=
		  "__kernel void symuluj(__global const float *wagi,__global const float *wejscia,__global float *preWyjscia,const int ilPolaczen) {"
		+ "		int polaczenie=get_global_id(1);"
		+ "		int index=get_global_id(0)*ilPolaczen+polaczenie;"
		+ "		"
		+ "		preWyjscia[index]=wejscia[polaczenie]*wagi[index];"
		+ "	}\n";
}