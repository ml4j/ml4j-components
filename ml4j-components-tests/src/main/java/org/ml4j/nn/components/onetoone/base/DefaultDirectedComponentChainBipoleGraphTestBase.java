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
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of ManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DefaultDirectedComponentChainBipoleGraphTestBase {

	
	@Mock
	protected NeuronsActivation mockInputActivation;
	
	@Mock
	protected NeuronsActivation mockNeuronsActivation1;
	
	@Mock
	protected MatrixFactory mockMatrixFactory;
	
	@Mock
	protected NeuronsActivation mockNeuronsActivation2;
	
	@Mock
	protected NeuronsActivation mockNeuronsActivation3;

	@Mock
	protected NeuronsActivation mockNeuronsActivation4;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	private DefaultChainableDirectedComponentActivation mockComponent1Activation;
	
	@Mock
	protected DefaultChainableDirectedComponentActivation mockComponent2Activation;
	
	@Mock
	protected DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, DirectedComponentsContext> mockComponent1;

	@Mock
	protected DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, DirectedComponentsContext> mockComponent2;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(mockMatrixFactory);
	    
	    Mockito.when(mockInputActivation.getFeatureCount()).thenReturn(10);
	    Mockito.when(mockInputActivation.getExampleCount()).thenReturn(32);
	    Mockito.when(mockInputActivation.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    
	    
	    Mockito.when(mockNeuronsActivation1.getFeatureCount()).thenReturn(100);
	    Mockito.when(mockNeuronsActivation1.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation1.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    Mockito.when(mockNeuronsActivation2.getFeatureCount()).thenReturn(200);
	    Mockito.when(mockNeuronsActivation2.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation2.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    Mockito.when(mockNeuronsActivation3.getFeatureCount()).thenReturn(300);
	    Mockito.when(mockNeuronsActivation3.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation3.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    Mockito.when(mockNeuronsActivation4.getFeatureCount()).thenReturn(400);
	    Mockito.when(mockNeuronsActivation4.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation4.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    
		Mockito.when(mockComponent1.forwardPropagate(Mockito.eq(mockNeuronsActivation1), Mockito.any())).thenReturn(mockComponent1Activation);
		Mockito.when(mockComponent2.forwardPropagate(Mockito.eq(mockNeuronsActivation2), Mockito.any())).thenReturn(mockComponent2Activation);
		Mockito.when(mockComponent1Activation.getOutput()).thenReturn(mockNeuronsActivation3);
		Mockito.when(mockComponent2Activation.getOutput()).thenReturn(mockNeuronsActivation4);

	}

	private DefaultDirectedComponentChainBipoleGraphBase createDefaultDirectedComponentChainBipoleGraph(List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		return createDefaultDirectedComponentChainBipoleGraphUnderTest(parallelComponents);
	}
		
	protected abstract DefaultDirectedComponentChainBipoleGraphBase createDefaultDirectedComponentChainBipoleGraphUnderTest(List<DefaultChainableDirectedComponent<?, ?>> components);

	@Test
	public void testConstruction() {	
		
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(mockComponents);
		Assert.assertNotNull(graph);
	}
	
	@Test
	public void testGetComponentType() {	
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(mockComponents);
		Assert.assertNotNull(graph);
		Assert.assertEquals(DirectedComponentType.COMPONENT_CHAIN_GRAPH, graph.getComponentType());
	}
	
	@Test
	public void testForwardPropagate() {	
				
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(mockComponents);
		Assert.assertNotNull(graph);
		
		DefaultDirectedComponentBipoleGraphActivation graphActivation =  graph.forwardPropagate(mockInputActivation, mockDirectedComponentsContext);
		Assert.assertNotNull(graphActivation);
		NeuronsActivation output = graphActivation.getOutput();
		Assert.assertNotNull(output);
		Assert.assertEquals(mockNeuronsActivation3.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(mockNeuronsActivation3.getFeatureOrientation(), output.getFeatureOrientation());
	}
	
	@Test
	public void testDup() {
				
		// TODO

	}
	

}
