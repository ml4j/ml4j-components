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
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of
 * OneToManyDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DifferentiableActivationFunctionComponentTestBase<L extends DifferentiableActivationFunctionComponent>
		extends TestBase {

	@Mock
	protected NeuronsActivation mockNeuronsActivation;

	// @Mock
	// protected DifferentiableActivationFunction mockActivationFunction;

	@Mock
	protected DifferentiableActivationFunctionActivation mockActivationFunctionActivation;

	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	private NeuronsActivation mockOutput;

	@Mock
	protected NeuronsActivationContext mockNeuronsActivationContext;
	
	@Mock
	protected DirectedComponentFactory mockDirectedComponentFactory;
	

	@Before
	public void setup() {
		MockitoAnnotations.initMocks(this);

		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

		Mockito.when(mockNeuronsActivation.getFeatureCount()).thenReturn(110);

		Mockito.when(mockNeuronsActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockNeuronsActivation.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

	}

	private L createDifferentiableActivationFunctionComponent(Neurons neurons,
			ActivationFunctionType activationFunctionType) {
		return createDifferentiableActivationFunctionComponentUnderTest(neurons, activationFunctionType);
	}

	protected abstract L createDifferentiableActivationFunctionComponentUnderTest(Neurons neurons,
			ActivationFunctionType activationFunctionType);

	@Test
	public void testConstruction() {

		ActivationFunctionType activationFunctionType = ActivationFunctionType
				.getBaseType(ActivationFunctionBaseType.RELU);

		Neurons neurons = new Neurons(100, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(
				neurons, activationFunctionType);
		Assert.assertNotNull(activationFunctionComponent);
	}

	@Test
	public void testGetComponentType() {
		Neurons neurons = new Neurons(100, false);

		ActivationFunctionType activationFunctionType = ActivationFunctionType
				.getBaseType(ActivationFunctionBaseType.RELU);

		// TODO THUR
		// Mockito.when(mockActivationFunction.getActivationFunctionType()).thenReturn(ActivationFunctionType.createCustomBaseType("DUMMY"));

		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(
				neurons, activationFunctionType);
		Assert.assertNotNull(activationFunctionComponent);
		Assert.assertEquals(NeuralComponentBaseType.ACTIVATION_FUNCTION,
				activationFunctionComponent.getComponentType().getBaseType());
	}

	@Test
	public void testForwardPropagate() {

		ActivationFunctionType activationFunctionType = ActivationFunctionType
				.getBaseType(ActivationFunctionBaseType.RELU);

		Mockito.when(mockOutput.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutput.getFeatureCount()).thenReturn(110);
		Mockito.when(mockOutput.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(mockOutput);

		// TODO THUR
		// Mockito.when(mockActivationFunction.activate(mockNeuronsActivation,
		// mockNeuronsActivationContext)).thenReturn(mockActivationFunctionActivation);

		Neurons neurons = new Neurons(110, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(
				neurons, activationFunctionType);
		Assert.assertNotNull(activationFunctionComponent);
		DifferentiableActivationFunctionComponentActivation activation = activationFunctionComponent
				.forwardPropagate(mockNeuronsActivation, mockNeuronsActivationContext);
		Assert.assertNotNull(activation);
		NeuronsActivation output = activation.getOutput();
		Assert.assertNotNull(output);

		Assert.assertFalse(output.getExampleCount() == 0);
		Assert.assertFalse(output.getFeatureCount() == 0);

		Assert.assertEquals(neurons.getNeuronCountExcludingBias(), output.getFeatureCount());
		Assert.assertEquals(mockNeuronsActivation.getExampleCount(), output.getExampleCount());

		// Assert.assertSame(mockOutput, output);

	}

	@Test
	public void testDup() {

		ActivationFunctionType activationFunctionType = ActivationFunctionType
				.getBaseType(ActivationFunctionBaseType.RELU);

		Mockito.when(mockOutput.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutput.getFeatureCount()).thenReturn(110);
		Mockito.when(mockOutput.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(mockOutput);

		// TODO THUR

		// Mockito.when(mockActivationFunction.getActivationFunctionType()).thenReturn(ActivationFunctionType.createCustomBaseType("DUMMY"));

		// Mockito.when(mockActivationFunction.activate(mockNeuronsActivation,
		// mockNeuronsActivationContext)).thenReturn(mockActivationFunctionActivation);

		Neurons neurons = new Neurons(110, false);
		DifferentiableActivationFunctionComponent activationFunctionComponent = createDifferentiableActivationFunctionComponent(
				neurons, activationFunctionType);
		DifferentiableActivationFunctionComponent dupComponent = activationFunctionComponent.dup(mockDirectedComponentFactory);

		DifferentiableActivationFunctionComponentActivation dupActivation = dupComponent
				.forwardPropagate(mockNeuronsActivation, mockNeuronsActivationContext);
		NeuronsActivation output = dupActivation.getOutput();
		Assert.assertNotNull(output);
		Assert.assertNotNull(dupComponent);
		Assert.assertEquals(NeuralComponentBaseType.ACTIVATION_FUNCTION, dupComponent.getComponentType().getBaseType());
	}

}
