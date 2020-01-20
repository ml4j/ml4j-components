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
package org.ml4j.nn.components.onetomany.base;

import java.util.List;
import java.util.function.IntSupplier;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
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
public abstract class OneToManyDirectedComponentTestBase extends TestBase {

	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;

	@Before
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
	}

	private OneToManyDirectedComponent<?> createOneToManyDirectedAxonsComponent(IntSupplier targetComponentsCount) {
		return createOneToManyDirectedComponentUnderTest(targetComponentsCount);
	}

	protected abstract OneToManyDirectedComponent<?> createOneToManyDirectedComponentUnderTest(
			IntSupplier targetComponentsCount);

	@Test
	public void testConstruction() {
		OneToManyDirectedComponent<?> oneToManyDirectedComponent = createOneToManyDirectedAxonsComponent(() -> 3);
		Assert.assertNotNull(oneToManyDirectedComponent);
	}

	@Test
	public void testGetComponentType() {
		OneToManyDirectedComponent<?> oneToManyDirectedComponent = createOneToManyDirectedAxonsComponent(() -> 3);
		Assert.assertEquals(NeuralComponentBaseType.ONE_TO_MANY,
				oneToManyDirectedComponent.getComponentType().getBaseType());
	}

	@Test
	public void testForwardPropagate() {

		int targetComponentCount = (int) (100 * Math.random());

		NeuronsActivation mockNeuronsActivation = MockTestData.mockNeuronsActivation(100, 32);

		OneToManyDirectedComponent<?> oneToManyDirectedComponent = createOneToManyDirectedAxonsComponent(
				() -> targetComponentCount);
		OneToManyDirectedComponentActivation activation = oneToManyDirectedComponent
				.forwardPropagate(mockNeuronsActivation, mockDirectedComponentsContext);
		Assert.assertNotNull(activation);
		List<NeuronsActivation> outputs = activation.getOutput();
		Assert.assertNotNull(outputs);
		Assert.assertEquals(targetComponentCount, outputs.size());
		outputs.forEach(o -> Assert.assertNotEquals(0, o.getFeatureCount()));
		outputs.forEach(o -> Assert.assertNotEquals(0, o.getExampleCount()));
		outputs.forEach(o -> Assert.assertNotNull(o.getFeatureOrientation()));

		outputs.forEach(o -> Assert.assertEquals(mockNeuronsActivation.getFeatureCount(), o.getFeatureCount()));
		outputs.forEach(o -> Assert.assertEquals(mockNeuronsActivation.getExampleCount(), o.getExampleCount()));
		outputs.forEach(
				o -> Assert.assertEquals(mockNeuronsActivation.getFeatureOrientation(), o.getFeatureOrientation()));

	}

	@Test
	public void testDup() {

		int targetComponentCount = (int) (100 * Math.random());

		OneToManyDirectedComponent<?> oneToManyDirectedComponent = createOneToManyDirectedAxonsComponent(
				() -> targetComponentCount);

		OneToManyDirectedComponent<?> dupComponent = oneToManyDirectedComponent.dup();

		NeuronsActivation mockNeuronsActivation = MockTestData.mockNeuronsActivation(100, 32);

		OneToManyDirectedComponentActivation dupActivation = dupComponent.forwardPropagate(mockNeuronsActivation,
				mockDirectedComponentsContext);
		List<NeuronsActivation> outputs = dupActivation.getOutput();
		Assert.assertNotNull(outputs);
		Assert.assertEquals(targetComponentCount, outputs.size());
		Assert.assertNotNull(dupComponent);
		Assert.assertEquals(NeuralComponentBaseType.ONE_TO_MANY,
				oneToManyDirectedComponent.getComponentType().getBaseType());
	}

}
