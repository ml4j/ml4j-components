package org.ml4j.nn.activationfunctions.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionTestBase extends TestBase {
	
	@Mock
	private NeuronsActivationContext context;
	
	@Mock
	private DifferentiableActivationFunctionActivation mockActivationFunctionActivation;
	
	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(context.getMatrixFactory()).thenReturn(matrixFactory);
	}

	@Test
	public void testConstruction() {
		
		Neurons leftNeurons = new Neurons1D(100, false);
		Neurons rightNeurons = new Neurons1D(100, false);

		DifferentiableActivationFunction activationFunction = 
				createDifferentiableActivationFunctionUnderTest(leftNeurons, rightNeurons);
		
		Assert.assertNotNull(activationFunction);
		Assert.assertNotNull(activationFunction.getActivationFunctionType());
		Assert.assertNotNull(activationFunction.getActivationFunctionType());
	}
	
	@Test
	public void testActivation() {
		
		Neurons leftNeurons = new Neurons1D(100, false);
		Neurons rightNeurons = new Neurons1D(100, false);

		DifferentiableActivationFunction activationFunction = 
				createDifferentiableActivationFunctionUnderTest(leftNeurons, rightNeurons);
		
		Assert.assertNotNull(activationFunction);
		
		NeuronsActivation inputActivation = createNeuronsActivation(100, 32);
		
		DifferentiableActivationFunctionActivation outputActivation = activationFunction.activate(inputActivation, context);
		
		Assert.assertNotNull(outputActivation);
		
		Assert.assertNotNull(outputActivation.getActivationFunction());
		
		Assert.assertNotNull(outputActivation.getInput());

		Assert.assertSame(inputActivation, outputActivation.getInput());
		
		Assert.assertNotNull(outputActivation.getOutput());

		Assert.assertEquals(32, outputActivation.getOutput().getExampleCount());
		Assert.assertEquals(100, outputActivation.getOutput().getFeatureCount());

	}
	
	@Test
	public void testActivationGradient() {
		
		Neurons leftNeurons = new Neurons1D(100, false);
		Neurons rightNeurons = new Neurons1D(100, false);

		DifferentiableActivationFunction activationFunction = 
				createDifferentiableActivationFunctionUnderTest(leftNeurons, rightNeurons);
		
		Assert.assertNotNull(activationFunction);
		
		NeuronsActivation inputActivation = createNeuronsActivation(100, 32);
		NeuronsActivation outputActivation = createNeuronsActivation(100, 32);

		Mockito.when(mockActivationFunctionActivation.getInput()).thenReturn(inputActivation);
		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(outputActivation);

		
		NeuronsActivation gradientActivation = activationFunction.activationGradient(mockActivationFunctionActivation, context);
		
		Assert.assertNotNull(gradientActivation);

		Assert.assertEquals(32, gradientActivation.getExampleCount());
		Assert.assertEquals(100, gradientActivation.getFeatureCount());

	}
	
	
	protected abstract DifferentiableActivationFunction createDifferentiableActivationFunctionUnderTest(Neurons leftNeurons, Neurons rightNeurons);

}
