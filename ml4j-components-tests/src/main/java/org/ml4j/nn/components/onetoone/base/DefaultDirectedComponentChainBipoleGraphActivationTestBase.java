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
package org.ml4j.nn.components.onetoone.base;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DefaultDirectedComponentChainBipoleGraphActivationTestBase extends TestBase {
	
	@Mock
	protected AxonsContext mockAxonsContext;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	protected DefaultDirectedComponentBipoleGraph mockBipoleGraph;
	
	protected DirectedComponentGradient<NeuronsActivation> mockInboundGradient;

	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
	    this.mockInboundGradient = MockTestData.mockComponentGradient(110, 32, this);

	}

	private DefaultDirectedComponentBipoleGraphActivation createDefaultDirectedComponentChainBipoleGraphActivation(DefaultDirectedComponentBipoleGraph bipoleGraph, NeuronsActivation output) {
		return createDefaultDirectedComponentChainBipoleGraphActivationUnderTest(bipoleGraph, output);
	}
		
	protected abstract DefaultDirectedComponentBipoleGraphActivation createDefaultDirectedComponentChainBipoleGraphActivationUnderTest(DefaultDirectedComponentBipoleGraph bipoleGraph, NeuronsActivation output);
	@Test
	public void testConstruction() {
		NeuronsActivation mockOutputActivation = MockTestData.mockNeuronsActivation(110, 32);

		DefaultDirectedComponentBipoleGraphActivation bipoleGraphActivation = createDefaultDirectedComponentChainBipoleGraphActivation(mockBipoleGraph, mockOutputActivation);
		Assert.assertNotNull(bipoleGraphActivation);
	}
	
	@Test
	public void testGetOutput() {
		
		NeuronsActivation mockOutputActivation = MockTestData.mockNeuronsActivation(110, 32);
				
		DefaultDirectedComponentBipoleGraphActivation bipoleGraphActivation = createDefaultDirectedComponentChainBipoleGraphActivation(mockBipoleGraph, mockOutputActivation);
		Assert.assertNotNull(bipoleGraphActivation);
		Assert.assertNotNull(bipoleGraphActivation.getOutput());
		Assert.assertSame(mockOutputActivation, bipoleGraphActivation.getOutput());
	}
	
	@Test
	public void testBackPropagate() {
		
		NeuronsActivation mockOutputActivation = MockTestData.mockNeuronsActivation(110, 32);
			
		Mockito.when(mockBipoleGraph.getInputNeurons()).thenReturn(new Neurons(100, false));
		Mockito.when(mockBipoleGraph.getOutputNeurons()).thenReturn(new Neurons(110, false));
	
		DefaultDirectedComponentBipoleGraphActivation bipoleGraphActivation = createDefaultDirectedComponentChainBipoleGraphActivation(mockBipoleGraph, mockOutputActivation);
		
		Assert.assertNotNull(bipoleGraphActivation);
		
		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = bipoleGraphActivation.backPropagate(mockInboundGradient);
		
		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());

		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);

		Assert.assertSame(mockBipoleGraph.getInputNeurons().getNeuronCountExcludingBias(), backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertSame(mockOutputActivation.getExampleCount(), backPropagatedGradient.getOutput().getExampleCount());

	}


}
