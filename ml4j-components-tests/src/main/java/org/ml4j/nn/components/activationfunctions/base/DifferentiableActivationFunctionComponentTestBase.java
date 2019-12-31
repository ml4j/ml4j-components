/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.activationfunctions.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of OneToManyDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DifferentiableActivationFunctionComponentTestBase extends TestBase {

	@Mock
	private NeuronsActivation mockNeuronsActivation;
	
	@Mock
	protected DifferentiableActivationFunction mockActivationFunction;
	
	
	@Mock
	protected DifferentiableActivationFunctionActivation mockActivationFunctionActivation;
	
	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	private NeuronsActivation mockOutput;
	
	
	@Mock
	private NeuronsActivationContext mockNeuronsActivationContext;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

	    Mockito.when(mockNeuronsActivation.getFeatureCount()).thenReturn(110);
	    
	    Mockito.when(mockNeuronsActivation.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

	}

	private DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons, DifferentiableActivationFunction activationFunction) {
		return createDifferentiableActivationFunctionComponentUnderTest(neurons, activationFunction);
	}
		
	protected abstract DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponentUnderTest(Neurons neurons, DifferentiableActivationFunction activationFunction);

	@Test
	public void testConstruction() {	
		Neurons neurons = new Neurons(100, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(neurons, mockActivationFunction);
		Assert.assertNotNull(activationFunctionComponent);
	}
	
	@Test
	public void testGetComponentType() {	
		Neurons neurons = new Neurons(100, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(neurons, mockActivationFunction);
		Assert.assertNotNull(activationFunctionComponent);
		Assert.assertEquals(DirectedComponentType.ACTIVATION_FUNCTION, activationFunctionComponent.getComponentType());
	}
	
	@Test
	public void testForwardPropagate() {	
		
		Mockito.when(mockOutput.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutput.getFeatureCount()).thenReturn(110);
		Mockito.when(mockOutput.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(mockOutput);
		
		Mockito.when(mockActivationFunction.activate(mockNeuronsActivation, mockNeuronsActivationContext)).thenReturn(mockActivationFunctionActivation);
		
		Neurons neurons = new Neurons(110, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(neurons, mockActivationFunction);
		Assert.assertNotNull(activationFunctionComponent);
		DifferentiableActivationFunctionComponentActivation activation =  activationFunctionComponent.forwardPropagate(mockNeuronsActivation, mockNeuronsActivationContext);
		Assert.assertNotNull(activation);
		NeuronsActivation output = activation.getOutput();
		Assert.assertNotNull(output);
		
		Assert.assertFalse(output.getExampleCount() == 0);
		Assert.assertFalse(output.getFeatureCount() == 0);

		Assert.assertEquals(neurons.getNeuronCountExcludingBias(), output.getFeatureCount());
		Assert.assertEquals(mockNeuronsActivation.getExampleCount(), output.getExampleCount());

		//Assert.assertSame(mockOutput, output);

	}
	
	@Test
	public void testDup() {
		
		Mockito.when(mockOutput.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutput.getFeatureCount()).thenReturn(110);
		Mockito.when(mockOutput.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(mockOutput);
		
		Mockito.when(mockActivationFunction.activate(mockNeuronsActivation, mockNeuronsActivationContext)).thenReturn(mockActivationFunctionActivation);
	
		
		Neurons neurons = new Neurons(110, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(neurons, mockActivationFunction);
		DifferentiableActivationFunctionComponent dupComponent = activationFunctionComponent.dup();
		
		DifferentiableActivationFunctionComponentActivation dupActivation =  dupComponent.forwardPropagate(mockNeuronsActivation, mockNeuronsActivationContext);
		NeuronsActivation output = dupActivation.getOutput();
		Assert.assertNotNull(output);
		Assert.assertNotNull(dupComponent);
		Assert.assertEquals(DirectedComponentType.ACTIVATION_FUNCTION, dupComponent.getComponentType());
	}
	

}
