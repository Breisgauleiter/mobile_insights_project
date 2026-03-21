/**
 * Results screen — displays highlight detection results for a video.
 */
import React, {useCallback, useState} from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import type {RootStackParamList} from '../../App';
import type {Highlight} from '../types';
import {fetchResults} from '../services/api';

type Props = NativeStackScreenProps<RootStackParamList, 'Results'>;

function formatTimestamp(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

export default function ResultsScreen({route}: Props) {
  const {id} = route.params;
  const [status, setStatus] = useState<string>('loading');
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await fetchResults(id);
      setStatus(data.status);
      setHighlights(data.highlights);
      setError(data.error ?? null);
    } catch (err) {
      setStatus('error');
      setError((err as Error).message);
    } finally {
      setRefreshing(false);
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  const onRefresh = () => {
    setRefreshing(true);
    load();
  };

  if (status === 'loading') {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6c5ce7" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.filename} numberOfLines={1}>
          {id}
        </Text>
        <Text style={[styles.status, statusColor(status)]}>{status}</Text>
      </View>

      {error && <Text style={styles.error}>{error}</Text>}

      {(status === 'pending' || status === 'processing') && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#fdcb6e" />
          <Text style={styles.processingText}>
            Processing video... Pull down to refresh.
          </Text>
        </View>
      )}

      {status === 'done' && (
        <FlatList
          data={highlights}
          keyExtractor={(_, i) => i.toString()}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
          ListEmptyComponent={
            <Text style={styles.empty}>No highlights detected</Text>
          }
          renderItem={({item, index}) => (
            <View style={styles.row}>
              <Text style={styles.index}>#{index + 1}</Text>
              <View style={styles.rowContent}>
                <Text style={styles.timestamp}>
                  {formatTimestamp(item.timestamp)}
                </Text>
                <Text style={styles.score}>
                  Score: {item.score.toFixed(1)}
                </Text>
              </View>
            </View>
          )}
        />
      )}
    </View>
  );
}

function statusColor(s: string) {
  switch (s) {
    case 'done':
      return {color: '#00b894'};
    case 'processing':
      return {color: '#fdcb6e'};
    case 'pending':
      return {color: '#74b9ff'};
    case 'error':
      return {color: '#d63031'};
    default:
      return {color: '#636e72'};
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e3a',
  },
  filename: {
    color: '#888',
    fontSize: 12,
    marginBottom: 4,
  },
  status: {
    fontSize: 16,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  error: {
    color: '#d63031',
    padding: 16,
    fontSize: 14,
  },
  processingText: {
    color: '#fdcb6e',
    marginTop: 16,
    fontSize: 14,
  },
  empty: {
    color: '#636e72',
    textAlign: 'center',
    marginTop: 40,
    fontSize: 16,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e3a',
  },
  index: {
    color: '#6c5ce7',
    fontSize: 14,
    fontWeight: 'bold',
    marginRight: 16,
    width: 30,
  },
  rowContent: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  timestamp: {
    color: '#fff',
    fontSize: 16,
  },
  score: {
    color: '#888',
    fontSize: 14,
  },
});
