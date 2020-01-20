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

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DirectedAxonsComponentActivationTestBase extends TestBase {

	protected NeuronsActivation mockOutputActivation;

	protected NeuronsActivation mockInputActivation;

	protected NeuronsActivation mockOutputActivationRightToLeft;

	@Mock
	protected AxonsContext mockAxonsContext;

	@SuppressWarnings("rawtypes")
	@Mock
	protected Axons mockAxons;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	protected AxonsActivation mockAxonsActivation;

	@Mock
	protected AxonsActivation mockAxonsActivationRightToLeft;

	@SuppressWarnings("rawtypes")
	@Mock
	protected DirectedAxonsComponent mockAxonsComponent;

	@Before
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockInputActivation = createNeuronsActivation(100, 32);
		this.mockOutputActivation = createNeuronsActivation(110, 32);
		this.mockOutputActivationRightToLeft = createNeuronsActivation(100, 32);
	}

	private <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DirectedAxonsComponentActivation createDirectedAxonsComponentActivation(
			DirectedAxonsComponent<L, R, A> axonsComponent, AxonsActivation axonsActivation,
			AxonsContext axonsContext) {
		return createDirectedAxonsComponentActivationUnderTest(axonsComponent, axonsActivation, axonsContext);
	}

	protected abstract <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DirectedAxonsComponentActivation createDirectedAxonsComponentActivationUnderTest(
			DirectedAxonsComponent<L, R, A> axonsComponent, AxonsActivation axonsActivation, AxonsContext axonsContext);

	@Test
	public void testConstruction() {
		@SuppressWarnings("unchecked")
		DirectedAxonsComponentActivation directedAxonsComponentActivation = createDirectedAxonsComponentActivation(
				mockAxonsComponent, mockAxonsActivation, mockAxonsContext);
		Assert.assertNotNull(directedAxonsComponentActivation);
	}

	@Test
	public void testGetAxonsComponent() {
		@SuppressWarnings("unchecked")
		DirectedAxonsComponentActivation directedAxonsComponentActivation = createDirectedAxonsComponentActivation(
				mockAxonsComponent, mockAxonsActivation, mockAxonsContext);
		Assert.assertNotNull(directedAxonsComponentActivation);
		Assert.assertNotNull(directedAxonsComponentActivation.getAxonsComponent());
		Assert.assertSame(mockAxonsComponent, directedAxonsComponentActivation.getAxonsComponent());
	}

	@Test
	public void testGetOutput() {

		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(mockOutputActivation);

		@SuppressWarnings("unchecked")
		DirectedAxonsComponentActivation directedAxonsComponentActivation = createDirectedAxonsComponentActivation(
				mockAxonsComponent, mockAxonsActivation, mockAxonsContext);
		Assert.assertNotNull(directedAxonsComponentActivation);
		Assert.assertNotNull(directedAxonsComponentActivation.getOutput());
		Assert.assertSame(mockOutputActivation, directedAxonsComponentActivation.getOutput());
	}

	@SuppressWarnings("unchecked")
	@Test
	public void testBackPropagate() {

		DirectedComponentGradient<NeuronsActivation> mockInboundGradient = MockTestData.mockComponentGradient(110, 32,
				this);

		Mockito.when(mockAxons.getLeftNeurons()).thenReturn(new Neurons(100, false));
		Mockito.when(mockAxons.getRightNeurons()).thenReturn(new Neurons(110, false));

		Mockito.when(mockAxonsActivation.getPostDropoutOutput()).thenReturn(mockOutputActivation);
		Mockito.when(mockAxonsActivation.getPostDropoutInput()).thenReturn(() -> mockInputActivation);

		Mockito.when(mockAxonsActivation.getAxons()).thenReturn(mockAxons);
		Mockito.when(mockAxonsComponent.getAxons()).thenReturn(mockAxons);

		Mockito.when(mockAxons.pushRightToLeft(mockInboundGradient.getOutput(), mockAxonsActivation, mockAxonsContext))
				.thenReturn(mockAxonsActivationRightToLeft);
		Mockito.when(mockAxonsActivationRightToLeft.getPostDropoutOutput()).thenReturn(mockOutputActivationRightToLeft);
		Mockito.when(mockAxonsActivationRightToLeft.getPostDropoutInput())
				.thenReturn(() -> mockOutputActivationRightToLeft);

		DirectedAxonsComponentActivation directedAxonsComponentActivation = createDirectedAxonsComponentActivation(
				mockAxonsComponent, mockAxonsActivation, mockAxonsContext);
		Assert.assertNotNull(directedAxonsComponentActivation);

		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = directedAxonsComponentActivation
				.backPropagate(mockInboundGradient);

		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());

		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);

		Assert.assertSame(mockOutputActivationRightToLeft.getFeatureCount(),
				backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertSame(mockOutputActivationRightToLeft.getExampleCount(),
				backPropagatedGradient.getOutput().getExampleCount());

	}

}
