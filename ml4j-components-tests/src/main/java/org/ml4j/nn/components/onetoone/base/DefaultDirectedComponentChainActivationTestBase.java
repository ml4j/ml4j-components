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
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DefaultDirectedComponentChainActivationTestBase extends TestBase {

	@Mock
	protected AxonsContext mockAxonsContext;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@SuppressWarnings("rawtypes")
	@Mock
	protected DirectedAxonsComponent mockAxonsComponent;

	@Mock
	protected DefaultDirectedComponentChain mockComponentChain;

	@Before
	public void setup() {
		MockitoAnnotations.initMocks(this);

		// Create the mock chain, with 100 input neurons and 110 output neurons.
		Mockito.when(mockComponentChain.getInputNeurons()).thenReturn(new Neurons(100, false));
		Mockito.when(mockComponentChain.getOutputNeurons()).thenReturn(new Neurons(110, false));

		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

	}

	private DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivation(
			DefaultDirectedComponentChain componentChain,
			List<DefaultChainableDirectedComponentActivation> activations) {
		return createDefaultDirectedComponentChainActivationUnderTest(componentChain, activations);
	}

	protected abstract DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivationUnderTest(
			DefaultDirectedComponentChain componentChain,
			List<DefaultChainableDirectedComponentActivation> activations);

	@Test
	public void testConstruction() {

		int exampleCount = 32;

		// Create the mock activations created by 2 components within the mock chain
		DefaultChainableDirectedComponentActivation mockChainableActivation1 = MockTestData.mockComponentActivation(100,
				105, exampleCount, this);
		DefaultChainableDirectedComponentActivation mockChainableActivation2 = MockTestData.mockComponentActivation(105,
				110, exampleCount, this);

		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(
				mockComponentChain, Arrays.asList(mockChainableActivation1, mockChainableActivation2));

		Assert.assertNotNull(componentChainActivation);
	}

	@Test
	public void testGetOutput() {

		int exampleCount = 32;

		// Create the mock chain activations
		DefaultChainableDirectedComponentActivation mockChainableActivation1 = MockTestData.mockComponentActivation(100,
				105, exampleCount, this);
		DefaultChainableDirectedComponentActivation mockChainableActivation2 = MockTestData.mockComponentActivation(105,
				110, exampleCount, this);

		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(
				mockComponentChain, Arrays.asList(mockChainableActivation1, mockChainableActivation2));
		Assert.assertNotNull(componentChainActivation);
		Assert.assertNotNull(componentChainActivation.getOutput());
		Assert.assertSame(mockChainableActivation2.getOutput(), componentChainActivation.getOutput());
	}

	@Test
	public void testGetActivations() {

		int exampleCount = 32;

		// Create the mock chain activations
		DefaultChainableDirectedComponentActivation mockChainableActivation1 = MockTestData.mockComponentActivation(100,
				105, exampleCount, this);
		DefaultChainableDirectedComponentActivation mockChainableActivation2 = MockTestData.mockComponentActivation(105,
				110, exampleCount, this);

		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(
				mockComponentChain, Arrays.asList(mockChainableActivation1, mockChainableActivation2));

		Assert.assertNotNull(componentChainActivation);
		Assert.assertNotNull(componentChainActivation.getActivations());
		Assert.assertFalse(componentChainActivation.getActivations().isEmpty());
		Assert.assertEquals(2, componentChainActivation.getActivations().size());
		Assert.assertSame(mockChainableActivation1, componentChainActivation.getActivations().get(0));
		Assert.assertSame(mockChainableActivation2, componentChainActivation.getActivations().get(1));

	}

	@Test
	public void testBackPropagate() {

		int exampleCount = 32;

		// Create the mock activations created by 2 components within the mock chain
		DefaultChainableDirectedComponentActivation mockChainableActivation1 = MockTestData.mockComponentActivation(100,
				105, exampleCount, this);
		DefaultChainableDirectedComponentActivation mockChainableActivation2 = MockTestData.mockComponentActivation(105,
				110, exampleCount, this);

		// Create the chain activation under test, from the mock chain and mock
		// activations.
		DefaultDirectedComponentChainActivation componentChainActivation = createDefaultDirectedComponentChainActivation(
				mockComponentChain, Arrays.asList(mockChainableActivation1, mockChainableActivation2));
		Assert.assertNotNull(componentChainActivation);

		// Create a mock in-bound gradient
		DirectedComponentGradient<NeuronsActivation> mockInboundGradient = MockTestData.mockComponentGradient(110, 32,
				this);

		// Back propagate the in-bound gradient through the chain activation
		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = componentChainActivation
				.backPropagate(mockInboundGradient);

		// Assertions
		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());

		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);

		Assert.assertSame(mockComponentChain.getInputNeurons().getNeuronCountExcludingBias(),
				backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertSame(exampleCount, backPropagatedGradient.getOutput().getExampleCount());
	}

}
