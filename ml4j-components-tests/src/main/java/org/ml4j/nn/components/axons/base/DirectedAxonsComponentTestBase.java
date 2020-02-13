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
package org.ml4j.nn.components.axons.base;

import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DirectedAxonsComponentTestBase extends TestBase {

	protected NeuronsActivation mockInputActivation;

	protected NeuronsActivation mockOutputActivation;

	@Mock
	protected AxonsContext mockAxonsContext;

	@Mock
	protected Axons<Neurons, Neurons, ?> mockAxons;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	protected DirectedComponentFactory mockDirectedComponentFactory;
	

	@Before
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockInputActivation = createNeuronsActivation(100, 32);
		this.mockOutputActivation = createNeuronsActivation(120, 32);

	}

	@SuppressWarnings("unchecked")
	private <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(
			L leftNeurons, R rightNeurons) {
		Mockito.when(mockAxons.getLeftNeurons()).thenReturn(leftNeurons);
		Mockito.when(mockAxons.getRightNeurons()).thenReturn(rightNeurons);
		Mockito.when(mockAxons.getAxonsType()).thenReturn(AxonsType.createCustomBaseType("someCustomType"));

		return createDirectedAxonsComponentUnderTest((Axons<L, R, ?>) mockAxons);
	}

	protected abstract <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponentUnderTest(
			Axons<L, R, ?> axons);

	@Test
	public void testConstruction() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);
		Assert.assertNotNull(directedAxonsComponent);
	}

	@Test
	public void testGetComponentType() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);
		Assert.assertEquals(NeuralComponentBaseType.AXONS, directedAxonsComponent.getComponentType().getBaseType());
	}

	@Test
	public void testDecompose() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);

		List<DefaultChainableDirectedComponent<?, ?>> components = directedAxonsComponent.decompose();
		Assert.assertNotNull(components);
		Assert.assertEquals(1, components.size());
		Assert.assertNotNull(components.get(0));
		Assert.assertEquals(directedAxonsComponent, components.get(0));
	}

	@Test
	public void testForwardPropagate() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		Mockito.when(mockInputActivation.getFeatureCount()).thenReturn(leftNeurons.getNeuronCountExcludingBias());
		Mockito.when(mockInputActivation.getExampleCount()).thenReturn(32);
		Mockito.when(mockInputActivation.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);

		DirectedAxonsComponentActivation activation = directedAxonsComponent.forwardPropagate(mockInputActivation,
				mockAxonsContext);
		Assert.assertNotNull(activation);
		Assert.assertNotNull(activation.getOutput());
		Assert.assertEquals(rightNeurons.getNeuronCountExcludingBias(), activation.getOutput().getFeatureCount());
		Assert.assertEquals(32, activation.getOutput().getExampleCount());
		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
				activation.getOutput().getFeatureOrientation());

	}

	@Test
	public void testDup() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);

		DirectedAxonsComponent<?, ?, ?> dupComponent = directedAxonsComponent.dup(mockDirectedComponentFactory);
		Assert.assertNotNull(dupComponent);
		Assert.assertNotSame(directedAxonsComponent, dupComponent);

	}

	@Test
	public void testGetAxons() {
		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);

		Axons<?, ?, ?> axons = directedAxonsComponent.getAxons();
		Assert.assertNotNull(axons);
	}

	@Test
	public void testGetContext() {

		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(120, false);

		DirectedAxonsComponent<?, ?, ?> directedAxonsComponent = createDirectedAxonsComponent(leftNeurons,
				rightNeurons);

		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

		AxonsContext mockAxonsContext2 = Mockito.mock(AxonsContext.class);

		Mockito.when(mockDirectedComponentsContext.getContext(Mockito.same(directedAxonsComponent), Mockito.any()))
				.thenReturn(mockAxonsContext2);

		AxonsContext axonsContext = directedAxonsComponent.getContext(mockDirectedComponentsContext);
		Assert.assertNotNull(axonsContext);
		Assert.assertNotSame(mockAxonsContext, axonsContext);
		Assert.assertSame(mockAxonsContext2, axonsContext);

	}

}
