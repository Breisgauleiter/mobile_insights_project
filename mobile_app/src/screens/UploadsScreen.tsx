/**
 * Uploads list screen — shows all uploaded videos and their processing status.
 */
import React, {useCallback, useState} from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import type {RootStackParamList} from '../../App';
import type {UploadItem} from '../types';
import {fetchUploads} from '../services/api';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Uploads'>;
};

const STATUS_COLORS: Record<string, string> = {
  done: '#00b894',
  processing: '#fdcb6e',
  pending: '#74b9ff',
  error: '#d63031',
  unknown: '#636e72',
};

export default function UploadsScreen({navigation}: Props) {
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await fetchUploads();
      setUploads(data.uploads);
    } catch {
      // silently fail — list stays empty
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      load();
    }, [load]),
  );

  const onRefresh = () => {
    setRefreshing(true);
    load();
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6c5ce7" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={uploads}
        keyExtractor={(item) => item.filename}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={
          <Text style={styles.empty}>No uploads yet</Text>
        }
        renderItem={({item}) => (
          <TouchableOpacity
            style={styles.row}
            onPress={() =>
              navigation.navigate('Results', {id: item.filename})
            }>
            <View style={styles.rowContent}>
              <Text style={styles.filename} numberOfLines={1}>
                {item.filename}
              </Text>
              <View
                style={[
                  styles.badge,
                  {backgroundColor: STATUS_COLORS[item.status] ?? '#636e72'},
                ]}>
                <Text style={styles.badgeText}>{item.status}</Text>
              </View>
            </View>
          </TouchableOpacity>
        )}
      />
    </View>
  );
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
    backgroundColor: '#0f0f23',
  },
  empty: {
    color: '#636e72',
    textAlign: 'center',
    marginTop: 40,
    fontSize: 16,
  },
  row: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e3a',
  },
  rowContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  filename: {
    color: '#fff',
    fontSize: 13,
    flex: 1,
    marginRight: 12,
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  badgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
});
