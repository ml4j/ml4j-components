package org.ml4j.nn.definitions;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.sessions.Session;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class Component3DtoNon3DGraphDefinitionTestBase<T extends NeuralComponent, D extends Component3DtoNon3DGraphDefinition> {

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	protected AxonsContext mockAxonsContext;
		
	protected NeuralComponentFactory<T> neuralComponentFactory;
	
	protected abstract NeuralComponentFactory<T> createNeuralComponentFactory();
	
	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.neuralComponentFactory = createNeuralComponentFactory();
		Mockito.when(mockDirectedComponentsContext.getContext(Mockito.any(), Mockito.any())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withRegularisationLambda(Mockito.anyFloat())).thenReturn(mockAxonsContext);
		Mockito.when(mockAxonsContext.withFreezeOut(Mockito.anyBoolean())).thenReturn(mockAxonsContext);
	}
	
	protected abstract void runAssertionsOnCreatedComponentGraph(D graphDefinition, 
			InitialComponentsGraphBuilder<T>  componentGraph);
	
	protected abstract D createDefinitionToTest();
	
	protected abstract Session<T> createSession(NeuralComponentFactory<T> neuralComponentFactory, DirectedComponentsContext directedComponentsContext);

	@Test
	public void testComponentGraphCreation() {
	
		// Start new session, given the component factory and the runtime context.
		Session<T> session = createSession(neuralComponentFactory, mockDirectedComponentsContext);
		
		// Create the graph definition to test
		D graphDefinition = createDefinitionToTest();
		
		// Build a component graph, given this Session and the definition.
		InitialComponentsGraphBuilder<T> componentGraph = session.startWith(graphDefinition);
			
		// Assert that we now have a component graph.
		Assert.assertNotNull(componentGraph);
		
		// Run additional assertions
		runAssertionsOnCreatedComponentGraph(graphDefinition, componentGraph);
	}

}
