package org.ml4j.nn.activationfunctions.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionActivationTestBase extends TestBase {
	
	@Mock
	private NeuronsActivationContext context;
	
	@Mock
	private DifferentiableActivationFunction mockActivationFunction;
	
	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(context.getMatrixFactory()).thenReturn(matrixFactory);
	}

	@Test
	public void testConstruction() {
		
		NeuronsActivation input = createNeuronsActivation(100, 32);
		NeuronsActivation output = createNeuronsActivation(100, 32);
		
		DifferentiableActivationFunctionActivation activationFunctionActivation = 
				createDifferentiableActivationFunctionActivationUnderTest(mockActivationFunction, input, output);
		
		Assert.assertNotNull(activationFunctionActivation);
		Assert.assertNotNull(activationFunctionActivation.getActivationFunction());
		Assert.assertSame(mockActivationFunction, activationFunctionActivation.getActivationFunction());
		Assert.assertSame(input, activationFunctionActivation.getInput());
		Assert.assertSame(output, activationFunctionActivation.getOutput());

	}
	
	
	
	protected abstract DifferentiableActivationFunctionActivation createDifferentiableActivationFunctionActivationUnderTest(DifferentiableActivationFunction activationFunction,
			NeuronsActivation input, NeuronsActivation output);

}
