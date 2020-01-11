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
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionComponentActivationTestBase extends TestBase {
	
	@Mock
	protected AxonsContext mockAxonsContext;
	
	protected NeuronsActivation mockInputActivation;
	
	protected NeuronsActivation mockOutputActivation;
	
	protected DirectedComponentGradient<NeuronsActivation> mockInboundGradient;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	protected DifferentiableActivationFunctionComponent mockActivationFunctionComponent;
	
	@Mock
	protected DifferentiableActivationFunction mockActivationFunction;

	@Mock
	protected ManyToOneDirectedComponent<?> mockManyToOneDirectedComponent;
		
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(createMatrixFactory());
	    Mockito.when(mockActivationFunctionComponent.getActivationFunction()).thenReturn(mockActivationFunction);
	    this.mockInputActivation = createNeuronsActivation(110, 32);
	    this.mockOutputActivation = createNeuronsActivation(110, 32);
	    this.mockInboundGradient = MockTestData.mockComponentGradient(110, 32, this);
	}

	private DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivation(DifferentiableActivationFunctionComponent activationFunction, NeuronsActivation input, NeuronsActivation output) {
		return createDifferentiableActivationFunctionComponentActivationUnderTest(activationFunction, input, output);
	}
		
	protected abstract DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivationUnderTest(DifferentiableActivationFunctionComponent activationFunction, NeuronsActivation input, NeuronsActivation output);
	
	@Test
	public void testConstruction() {
		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(mockActivationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assert.assertNotNull(activation);
	}
	
	@Test
	public void testGetOutput() {
		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(mockActivationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assert.assertNotNull(activation);
		Assert.assertNotNull(activation.getOutput());
		Assert.assertSame(mockOutputActivation, activation.getOutput());
	}
	
	@Test
	public void testBackPropagate() {
		
		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(mockActivationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assert.assertNotNull(activation);
		
		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = activation.backPropagate(mockInboundGradient);
		
		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);

		Assert.assertSame(110, backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertSame(32, backPropagatedGradient.getOutput().getExampleCount());

	}


}
