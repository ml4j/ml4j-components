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
package org.ml4j.nn.components.manytoone;

import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;


/**
 * Unit test for DummyManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public class DummyManyToOneDirectedComponentTest extends ManyToOneDirectedComponentTestBase {

	@Mock
	private NeuronsActivation mockNeuronsActivation1;
	
	@Mock
	private NeuronsActivation mockNeuronsActivation2;
	
	
	@Override
	protected ManyToOneDirectedComponent<?> createManyToOneDirectedComponentUnderTest(
			PathCombinationStrategy pathCombinationStrategy) {
		return new DummyManyToOneDirectedComponent(pathCombinationStrategy);
	}

	@Override
	protected NeuronsActivation createNeuronsActivation1(int featureCount, int examples) {
		Mockito.when(mockNeuronsActivation1.getFeatureCount()).thenReturn(featureCount);
		Mockito.when(mockNeuronsActivation1.getExampleCount()).thenReturn(examples);
		Mockito.when(mockNeuronsActivation1.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		return mockNeuronsActivation1;
	}
	
	@Override
	protected NeuronsActivation createNeuronsActivation2(int featureCount, int examples) {
		Mockito.when(mockNeuronsActivation2.getFeatureCount()).thenReturn(featureCount);
		Mockito.when(mockNeuronsActivation2.getExampleCount()).thenReturn(examples);
		Mockito.when(mockNeuronsActivation2.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		return mockNeuronsActivation2;
	}

}
