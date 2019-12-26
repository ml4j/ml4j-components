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

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DefaultDirectedComponentChainActivationTestBase {

	@Mock
	protected NeuronsActivation mockOutputActivation;
	
	@Mock
	protected NeuronsActivation mockInputActivation;
	
	@Mock
	protected AxonsContext mockAxonsContext;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	protected MatrixFactory mockMatrixFactory;
	
	@Mock
	protected DirectedComponentGradient<NeuronsActivation> mockInboundGradient;
	
	@Mock
	protected NeuronsActivation mockGradientActivation;
	
	@SuppressWarnings("rawtypes")
	@Mock
	protected DirectedAxonsComponent mockAxonsComponent;
	
	@Mock
	protected DefaultDirectedComponentChain mockComponentChain;
	
	@Mock
	protected DefaultChainableDirectedComponentActivation mockChainable1;
	
	@Mock
	protected DefaultChainableDirectedComponentActivation mockChainable2;
	
	@Mock
	protected DirectedComponentGradient<NeuronsActivation> mockMiddleGradient;
	
	@Mock
	protected DirectedComponentGradient<NeuronsActivation> mockOutputGradient;
	
	@Mock
	protected NeuronsActivation mockOutputGradientActivation;
	
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(mockMatrixFactory);

	}

	private DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivation(DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations,  NeuronsActivation output) {
		return createDefaultDirectedComponentChainActivationUnderTest(componentChain, activations, output);
	}
		
	protected abstract DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivationUnderTest(
			DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations, NeuronsActivation output);
	@Test
	public void testConstruction() {
		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(mockComponentChain, Arrays.asList(mockChainable1, mockChainable2), mockOutputActivation);
		Assert.assertNotNull(componentChainActivation);
	}
	
	@Test
	public void testGetOutput() {
				
		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(mockComponentChain, Arrays.asList(mockChainable1, mockChainable2), mockOutputActivation);
		Assert.assertNotNull(componentChainActivation);
		Assert.assertNotNull(componentChainActivation.getOutput());
		Assert.assertSame(mockOutputActivation, componentChainActivation.getOutput());
	}
	
	@Test
	public void testGetActivations() {
						
		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(mockComponentChain, Arrays.asList(mockChainable1, mockChainable2), mockOutputActivation);
		Assert.assertNotNull(componentChainActivation);
		Assert.assertNotNull(componentChainActivation.getActivations());
		Assert.assertFalse(componentChainActivation.getActivations().isEmpty());
		Assert.assertEquals(2, componentChainActivation.getActivations().size());

	}
	
	@Test
	public void testBackPropagate() {
		
		Mockito.when(mockInboundGradient.getOutput()).thenReturn(mockGradientActivation);
		Mockito.when(mockGradientActivation.getFeatureCount()).thenReturn(110);
		Mockito.when(mockGradientActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockGradientActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockOutputActivation.getFeatureCount()).thenReturn(110);
		Mockito.when(mockOutputActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutputActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Mockito.when(mockComponentChain.getInputNeurons()).thenReturn(new Neurons(100, false));
		Mockito.when(mockComponentChain.getOutputNeurons()).thenReturn(new Neurons(110, false));


		Mockito.when(mockInputActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockInputActivation.getFeatureCount()).thenReturn(100);
		Mockito.when(mockInputActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		
		
		Mockito.when(mockChainable2.backPropagate(mockInboundGradient)).thenReturn(mockMiddleGradient);
		
		Mockito.when(mockChainable1.backPropagate(mockMiddleGradient)).thenReturn(mockOutputGradient);
		
		Mockito.when(mockOutputGradientActivation.getFeatureCount()).thenReturn(100);
		Mockito.when(mockOutputGradientActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockOutputGradientActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		Mockito.when(mockOutputGradient.getOutput()).thenReturn(mockOutputGradientActivation);
		

		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(mockComponentChain, Arrays.asList(mockChainable1, mockChainable2), mockOutputActivation);
		
		
		Assert.assertNotNull(componentChainActivation);
		
		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = componentChainActivation.backPropagate(mockInboundGradient);
		
		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());

		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);

		Assert.assertSame(mockComponentChain.getInputNeurons().getNeuronCountExcludingBias(), backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertSame(mockOutputActivation.getExampleCount(), backPropagatedGradient.getOutput().getExampleCount());

	}


}
