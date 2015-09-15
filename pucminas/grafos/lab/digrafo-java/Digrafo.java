/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos.
 */
import java.util.List;

/**
 * Classe que implementa um grafo direcionado, ou digrafo.
 */
public abstract class Digrafo {

   /**
    * Adiciona uma nova aresta direcionada do vertice u para v.
    */
	public abstract void adicionaAresta(int u, int v);

   /**
    * Remove uma nova aresta direcionada do vertice u para v, caso a mesma exista.
    */
	public abstract void removeAresta(int u, int v);

   /**
    * Verifica se existe uma aresta direcionada do vertice u para v.
    * @return true caso existir a aresta (u,v) ou false caso contrario.
    */
	public abstract boolean existeAresta(int u, int v);

	/**
	 * Obtem o numero de vertices do digrafo.
    * @return o numero de vertices.
	 */
	public abstract int numVertices();

	/**
	 * Obtem o numero de arestas direcionadas do digrafo.
    * @return o numero de arestas do digrafo.
	 */
	public abstract int numArestas();

	/**
	 * Obtem os vizinhos de um dado vertice u.
    * @return conjunto de vizinhos do vertice u.
	 */
	public abstract List<Integer> vizinhos(int u);

	/**
	 * Verifica se existe algum loop no Digrafo.
    * @return true caso existir algum loop em qualquer vertice ou false caso contrario.
	 */
	public abstract boolean existeLoop();

	/**
	 * Calcula o grau de saida do vertice u.
    * O grau de saida de um vertice u representa o numero de arestas direcionadas que possuem u como origem.
    * @return grau de saida do vertice u.
	 */
	public abstract int grauSaida(int u);

	/**
	 * Calcula o grau de entrada do vertice u.
    * O grau de entrada de um vertice u representa o numero de arestas direcionadas que possuem u como destino.
    * @return grau de entrada do vertice u.
	 */
	public abstract int grauEntrada(int u);

}
